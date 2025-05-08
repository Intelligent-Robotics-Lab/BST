import cv2, mediapipe as mp, threading, queue, time

# Constants for behavior detection
_ACTION_COOLDOWN = 12  # number of frames to wait before allowing the same action to be detected again
_NOD_DELTA       = 0.02  # vertical movement threshold to detect a nod
_OBJ_MATCH_MIN   = 15  # unused here, may be used for object matching elsewhere

class VisionDetector(threading.Thread):
    def __init__(self, out_queue: queue.Queue, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.out_queue  = out_queue
        self.stop_evt   = stop_evt
        self.cooldowns  = {"nod-head":0, "touch-head":0, "hands-up":0}
        self.prev_nose_y = None

        # initialize MediaPipe's Holistic model
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def _decay_cooldowns(self):
        for k in self.cooldowns:
            if self.cooldowns[k] > 0:
                self.cooldowns[k] -= 1

    # Main loop
    def run(self):
        cap = cv2.VideoCapture(1) # EXTERNAL MICROSOFT CAMERA
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            res = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            action = None
            # 1) hands‑up -------------------------------------------------
            if res.pose_landmarks and self.cooldowns["hands-up"]==0:
                lm = res.pose_landmarks.landmark
                
                # find shoulder and wrist locations
                l_sh, r_sh = lm[self.mp_holistic.PoseLandmark.LEFT_SHOULDER], \
                             lm[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                l_wr, r_wr = lm[self.mp_holistic.PoseLandmark.LEFT_WRIST], \
                             lm[self.mp_holistic.PoseLandmark.RIGHT_WRIST]
                
                # If both wrists are above their respective shoulders, send "hands-up" as the output, which is then stored in Main.py and fed to the LLM
                if l_wr.y < l_sh.y and r_wr.y < r_sh.y:
                    action = "hands-up"

            # 2) touch‑head ----------------------------------------------
            if not action and res.face_landmarks and self.cooldowns["touch-head"]==0:
                
                # any hand landmark inside a small box above the head?
                head_y = min(lm.y for lm in res.face_landmarks.landmark)
                
                if res.left_hand_landmarks or res.right_hand_landmarks:
                    hand = res.left_hand_landmarks or res.right_hand_landmarks
                    tip  = hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP]
                    if tip.y < head_y:  # if ANY part of the hand is above above top of face, we assume the interventionist is touching their head
                        action = "touch-head"

            # 3) nod‑head -----------------------------------------------
            if res.pose_landmarks and self.cooldowns["nod-head"]==0:
                nose_y = res.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].y
                if self.prev_nose_y is not None:
                    dy = nose_y - self.prev_nose_y
                    if dy >  _NOD_DELTA:   # going down
                        self._nod_state = "down"
                    elif dy < -_NOD_DELTA and getattr(self, "_nod_state",None)=="down":
                        action = "nod-head"
                        self._nod_state = "cool"
                self.prev_nose_y = nose_y

            if action:
                self.out_queue.put(action)
                self.cooldowns[action] = _ACTION_COOLDOWN

            self._decay_cooldowns()

            # comment the next two lines out if you want to hide the video preview
            cv2.imshow("Vision", frame)
            if cv2.waitKey(5)&0xFF==27: self.stop_evt.set()

        cap.release();  cv2.destroyAllWindows()
