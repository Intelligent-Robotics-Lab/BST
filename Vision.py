import cv2
import mediapipe as mp

labels = ['ball','book','car']
orb    = cv2.ORB_create(500)
bf     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
templates = {}
for lbl, fname in zip(labels, ['ball.jpg','book.jpg','car.jpg']):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img, None)
    templates[lbl] = des

mp_holistic = mp.solutions.holistic

OBJ_MATCH_THRESH   = 15    # min ORB matches to accept an object
HEAD_EXTENSION     = 0.5
HORIZ_MARGIN       = 0.1
HEAD_TOUCH_CD_INIT = 10
NOD_DELTA          = 0.02
NOD_CD_INIT        = 15
ARMS_UP_CD_INIT    = 10    # cooldown frames after a hands‑up detection

prev_nose_y    = None
nod_state      = "ready"
nod_cd         = 0
head_touch_cd  = 0
arms_up_cd     = 0

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = holistic.process(rgb)

        # 1) Object-in-hand
        obj_label = None
        hand = res.left_hand_landmarks or res.right_hand_landmarks
        if hand:
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            pad = 0.2
            x1n, x2n = max(0, min(xs)-pad), min(1, max(xs)+pad)
            y1n, y2n = max(0, min(ys)-pad), min(1, max(ys)+pad)
            x1, y1 = int(x1n*w), int(y1n*h)
            x2, y2 = int(x2n*w), int(y2n*h)
            roi = frame[y1:y2, x1:x2]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                kp2, des2 = orb.detectAndCompute(gray, None)
                if des2 is not None:
                    best, best_lbl = 0, None
                    for lbl, des1 in templates.items():
                        if des1 is None: continue
                        matches = bf.match(des1, des2)
                        good = [m for m in matches if m.distance < 60]
                        if len(good) > best:
                            best, best_lbl = len(good), lbl
                    if best > OBJ_MATCH_THRESH:
                        obj_label = best_lbl

        # 2) Top‑of‑head touch
        head_touch = False
        if hand and res.face_landmarks and head_touch_cd == 0:
            fx = [lm.x for lm in res.face_landmarks.landmark]
            fy = [lm.y for lm in res.face_landmarks.landmark]
            min_x, max_x = min(fx), max(fx)
            min_y, max_y = min(fy), max(fy)
            fw, fh = max_x-min_x, max_y-min_y

            rx1 = min_x - HORIZ_MARGIN*fw
            rx2 = max_x + HORIZ_MARGIN*fw
            ry2 = min_y
            ry1 = max(0.0, ry2 - HEAD_EXTENSION*fh)

            tip = hand.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            if rx1 <= tip.x <= rx2 and ry1 <= tip.y <= ry2:
                head_touch = True
                head_touch_cd = HEAD_TOUCH_CD_INIT

        if head_touch_cd > 0:
            head_touch_cd -= 1

        # 3) Hands‑up detection
        arms_up = False
        if res.pose_landmarks and arms_up_cd == 0:
            lm = res.pose_landmarks.landmark
            l_sh, r_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER], lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            l_wr, r_wr = lm[mp_holistic.PoseLandmark.LEFT_WRIST],    lm[mp_holistic.PoseLandmark.RIGHT_WRIST]
            if l_wr.y < l_sh.y and r_wr.y < r_sh.y:
                arms_up = True
                arms_up_cd = ARMS_UP_CD_INIT

        if arms_up_cd > 0:
            arms_up_cd -= 1

        # 4) Head‑nod detection
        nod_detect = False
        if res.pose_landmarks:
            nose_y = res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y
            if prev_nose_y is not None and nod_cd == 0:
                d = nose_y - prev_nose_y
                if nod_state == "ready" and d > NOD_DELTA:
                    nod_state = "down_detected"
                elif nod_state == "down_detected" and d < -NOD_DELTA:
                    nod_detect = True
                    nod_cd = NOD_CD_INIT
                    nod_state = "cooldown"
            prev_nose_y = nose_y

        if nod_cd > 0:
            nod_cd -= 1
            if nod_cd == 0:
                nod_state = "ready"

        # 5) Priority & output
        if nod_detect:
            action = "nod-head"
        elif head_touch:
            action = "touch-head"
        elif arms_up:
            action = "hands-up"
        elif obj_label:
            action = f"hold-{obj_label}"
        else:
            action = None

        if action:
            print(action)
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow("Detecting…", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
