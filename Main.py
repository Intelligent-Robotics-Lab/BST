import sys
import json
import time
import random
import requests
import queue, threading                     
import LookupTables
import os

from Vision import VisionDetector          
from langchain_openai import ChatOpenAI
from furhat_remote_api import FurhatRemoteAPI
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet


''' DTT session class. Instantiated for every new session. It handles:
- Initializing Furhat, NAO, and the LLM
- Running each trial
    - Capturing interventionist's instruction and consequence strategy
    - LLM call 1: determining which trial the session is in, given the interventionist's instruction
    - NAO response to the instruction
    - LLM call 2: testing the accuracy of the interventionist's responses/gestures
    - PDF generation
'''

class DTTSession:
    def __init__(self):
        
        # Connect to Furhat
        self.furhat = FurhatRemoteAPI("141.210.88.11")
        self.furhat.attend(user="CLOSEST")
        time.sleep(2)

        self.trials = LookupTables.trials  # lookup tables file
        self.start_time = time.time()  # program start time, for calculating trial lengths later
        self.DTT_state = "WELCOME"  # state variable (string) that holds which trial type (or the welcome/results states) that the program is in
        self.evaluation_results = []  # used for the PDF at the end
        self.gpt41 = self._initialize_gpt41()  # initializes the LLM and stores it in this variable to be called later

        # Initializes the vision thread to work with the Vision.py file.
        # Starts a queue and threading to allow for asynchronous detection of movements outside of the main loop.
        # I've found this is helpful for more natural turn-taking.
        self.vision_q   = queue.Queue()
        self.stop_evt   = threading.Event()
        self.vision_thr = VisionDetector(self.vision_q, self.stop_evt)
        self.vision_thr.start()

    # I have chosen GPT-4.1, as it was cheaper and had better reasoning than GPT-4o. In the future, I recommend we move to a local model for quicker outputs.
    # Don't forget to put in the API key! I removed it here for safety.
    def _initialize_gpt41(self):
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            max_retries=2,
            api_key=""
        )
    # It would be easy to have multiple model initializations here, and simply change the variable to a different model when invoking the LLM later.
    # I did this to test between 4o, 4o-mini, and 4.1 easily.
    # I noticed that, despite being smaller, the 4o-mini model did not yield significantly shorter invoke times, so I stuck with 4o and moved to 4.1 when it released.

    # Core session logic.
    # Currently, this is only the rehearsal stage.
    # What you saw in the videos of the other stages was hard-coded, which I've removed (as it isn't actually functional and was just for demonstration purposes).
    def run_session(self):

        time.sleep(5) # so I can run to the other side of the table after pressing play, lol

        # This loop acts as a way to run multiple trials, determined by the loop's bounds.
        # Currently, the program runs 3 trials (the upper bound is exclusive).
        # It may be useful to run a variable number of trials, set by a difficulty setting, or run an indefinite number of trials,
        # only ending when the interventionist says a key phrase.
        for trial_iteration in range(1, 4):
            print(f"\n=== Trial {trial_iteration} ===") # for debugging purposes

            # if you find changing Furhat's LED a useful indication of turn-taking, it can be changed like this:
            # self.set_led("red")

            # I have moved to NAO blinking three times, as I think it's more intuitive.
            # Ideally, there is no need for this, and the system is always listening.

            # On the first trial, have Furhat say "begin now"
            if trial_iteration == 1:
                self.furhat.say(text="Please begin now!")
                
                # I call sleep here because Furhat's .say function is asynchronous. 
                # You need to sleep the program for as long as it takes Furhat to say the given phrase.
                # Because of this, I'd recommend moving away from Furhat's ASR.
                time.sleep(2)  

            # at the beginning, I found it useful to indicate new trials with Furhat giving explicit instructions:
            #else:
                #self.furhat.say(text="Please provide your next instruction.")

            
            # ======== INSTRUCTION (SD) ====================================

            
            # clear any stored gestures so we only catch those made from now on
            while not self.vision_q.empty():
                _ = self.vision_q.get_nowait()

            # This is NAO's behavior for blinking three times, indicating that Furhat is listening. 
            # Yes, it's called "LightGreen", and I forgot to change the name to something like "BlinkThreeTimes"
            naoBehavior(".lastUploadedChoregrapheBehavior/LightGreen")

            # Listening with Furhat's Python remote API function
            instruction_result = self.furhat.listen()
            time.sleep(1)

            # this if-statement makes sure we safely extract the string from Furhat's ASR
            if hasattr(instruction_result, "message"):
                # if the returned object from Furhat's ASR has a "message" attribute, we know it's safe to extract a string
                bt_instruction = instruction_result.message
            elif isinstance(instruction_result, dict):
                # if it's a dictionary, it tries to get the key, and otherwise assigns an empty string.
                bt_instruction = instruction_result.get("message", "")
            else:
                # otherwise, it's just converted to a string
                bt_instruction = str(instruction_result)

            # if there were any gestures detected from the vision model during this time, they are assigned to the "vision_action" variable here to be fed into the LLM below.
            # this is how I detect "do this" instructions
            vision_action = "none"
            while not self.vision_q.empty():
                vision_action = self.vision_q.get_nowait()

            # -------- LLM (for instruction) ---------------------------------

            # Here is the prompt to the LLM that solely determines which state we're in, given the instruction from the interventionist.
            # It is fed the BT's instruction and any gesture (vision_action) that was detected.
            # It is also given the dictionary from LookupTables to select the most appropriate state. 
            # Remember that the dictionary can be expanded anytime to accommodate more actions!
            # This LLM invocation should only return a number from 1 to 13, each corresponding to a trial in the dictionary.
            # This is how we effectively select which state we're in.
            
            prompt = f"""
            You are an ABA DTT expert. A behavior technician is being trained to perform DTT with a child.
            Here is what the BT has said to the child as an instruction (Discriminative Stimulus, or SD): {bt_instruction}
            While speaking, the BT’s gesture‑tracking camera detected: {vision_action}
            If an imitation instruction is presented, but there is no gesture detected, please return "0"
            Here is a dictionary of all of the possible and ideal phrasing for the instructions: {self.trials}

            Your job is to scan the dictionary of possible trials and tell me which number trial the instruction BEST corresponds to.
            It's OK if the phrasing is slightly off and it is also OK if the child's name (Emily) is used.
            However, if the instruction absolutely does not match any of the trials, return "0".

            Respond with ONLY the trial number (e.g., "5"). No other words.
            """

            # Notice how I prompt it with "It's OK if the phrasing is slightly off"
            # I was hoping to explore some more fine-tuning with these prompting techniques. What does "slightly off" mean? Should we give examples to the LLM?
            # There is a lot to explore here, but for the most part, I found it effective in my limited testing.
            
            # invoke the LLM with the prompt above, otherwise throw an error.
            trial_number = "N/A"
            try:
                response      = self.gpt41.invoke([{"role": "system", "content": prompt}])
                trial_number  = response.content if hasattr(response, "content") else str(response)
                print("LLM Output:", trial_number) 
            except Exception as e:
                print("Error while processing instruction:", e)

            # ======== CHILD REACTION ===============================

            # Now that we know which trial we're in, we can have NAO react accordingly.
            # Right now, there is a 50/50 chance that NAO responds correctly/incorrectly.
            # The incorrect reactions are just something I chose that was different than the intended action.
            # Obviously, this should be expanded to add more nuance, like the "frustration model" that we discussed.
            
            reaction = random.randrange(1, 3)
            # reaction = 1: NAO will respond correctly
            # recation = 2: NAO will respond incorrectly

            child_response = "N/A"

            # Given the LLM's output number, we run the corresponding NAO behavior for that trial.
            # Also, I manually store the "child_response" as I found it helpful to use later when we invoke the LLM again.
            # When adding more trials to the dictionary, remember to also add them here in the case/switch.
            match trial_number:
                case "1":
                    naoBehavior(".lastUploadedChoregrapheBehavior/behavior_1")
                    child_response = "child excitedly says: 'I want stickers!'"
                case "2":
                    if reaction == 1:
                        naoBehavior("bststudy/Touch Head")
                        child_response = "child successfully touches their head"
                    elif reaction == 2:
                        naoBehavior("bststudy/Ball")
                        child_response = "child does NOT touch their head"
                case "3":
                    if reaction == 1:
                        child_response = "child successfully waves"
                        naoBehavior("bststudy/Wave Prompt")
                    elif reaction == 2:
                        child_response = "child does NOT wave"
                        naoBehavior("bststudy/Clap Prompt")
                case "4":
                    if reaction == 1:
                        naoBehavior("bststudy/Car")
                        child_response = "child successfully guesses the object is a toy car"
                    elif reaction == 2:
                        child_response = "child does NOT guess the object is a toy car"
                        naoBehavior("bststudy/Ball")
                case "5":
                    if reaction == 1:
                        child_response = "child successfully guesses the happy face"
                        naoBehavior("bststudy/Happy")
                    elif reaction == 2:
                        child_response = "child does NOT guess the happy face"
                        naoBehavior("bststudy/Angry")
                case "6":
                    if reaction == 1:
                        naoBehavior("bststudy/Arms Up Prompt")
                        child_response = "child successfully puts their arms up"
                    elif reaction == 2:
                        naoBehavior("bststudy/Clap Prompt")
                        child_response = "child does NOT put their arms up"
                case "7":
                    if reaction == 1:
                        naoBehavior("bststudy/Clap Prompt")
                        child_response = "child successfully claps their hands"
                    elif reaction == 2:
                        naoBehavior("bststudy/Touch Nose Prompt")
                        child_response = "child does NOT clap their hands"
                case "8":
                    if reaction == 1:
                        child_response = "child successfully guesses the object is a toy ball"
                        naoBehavior("bststudy/Ball")
                    elif reaction == 2:
                        child_response = "child does NOT guess the object is a toy ball"
                        naoBehavior("bststudy/Book")
                case "9":
                    if reaction == 1:
                        child_response = "child successfully guesses the sad face"
                        naoBehavior("bststudy/Sad")
                    elif reaction == 2:
                        child_response = "child does NOT guess the sad face"
                        naoBehavior("bststudy/Happy")
                case "10":
                    if reaction == 1:
                        child_response = "child successfully nods their head"
                        naoBehavior("bststudy/Nod Yes")
                    elif reaction == 2:
                        child_response = "child does NOT nod their head"
                        naoBehavior("bststudy/Clap Prompt")
                case "11":
                    if reaction == 1:
                        naoBehavior("bststudy/Touch Nose Prompt")
                        child_response = "child successfully touches their nose"
                    elif reaction == 2:
                        naoBehavior("bststudy/Arms Up Prompt")
                        child_response = "child does NOT touch their nose"
                case "12":
                    if reaction == 1:
                        child_response = "child successfully guesses the object is a book"
                        naoBehavior("bststudy/Book")
                    elif reaction == 2:
                        child_response = "child does NOT guess the object is a book"
                        naoBehavior("bststudy/Ball")
                case "13":
                    if reaction == 1:
                        child_response = "child successfully guesses the angry face"
                        naoBehavior("bststudy/Angry")
                    elif reaction == 2:
                        child_response = "child does NOT guess the angry face"   
                        naoBehavior("bststudy/Happy")
                case "0":
                    # NOT A VALID STATE PROVIDED BY BT
                    child_response = "Error: Not a valid state provided by the interventionist!"
                case _:
                    # NOT A VALID OUTPUT FROM LLM
                    child_response = "Error: The LLM's output was not readable!"     

            
            # ======== CONSEQUENCE STRATEGY ============================

            
            # Now we listen for the interventionist's reinforcement/error-correction to the child's reaction.
            # Currently, there is no prompt logic, as the intervention time/delay (zero-second, two-second) is difficult with Furhat's ASR.

            # These sleeps can be tuned. I actually don't remember why this one is 2.5 seconds. If you switch to a different ASR, these sleeps are unnecessary.
            time.sleep(2.5)

            # Blink three times, then listen
            naoBehavior(".lastUploadedChoregrapheBehavior/LightGreen")
            cs_result = self.furhat.listen()
            time.sleep(1)

            # Same logic as the last time
            if hasattr(cs_result, "message"):
                bt_cs = cs_result.message
            elif isinstance(cs_result, dict):
                bt_cs = cs_result.get("message", "")
            else:
                bt_cs = str(cs_result)

            # ======== LLM: evaluate full trial ============================

            # We now call the "evaluate_technician_action" function (below) and store the LLM's output in "result" to be appended to the PDF, 
            # and to eventually be used in the feedback stage (needs to be added).
            # Pass through the trial number, instruction, consequence strategy, child response, gesture, and the LLM's model.
            result = self.evaluate_technician_action(
                trial_number,
                bt_instruction,
                bt_cs,
                child_response,
                vision_action,
                self.gpt41
            )

            # For each trial, we create a new row in the PDF, but for now, we need to store it somehow
            self.evaluation_results.append(result)

            if "Trial Type" in result:
                self.DTT_state = result["Trial Type"]
            # Change the internal variable depending on which trial we were in. I stopped needing this variable, but it may be helpful as the program expands, so I kept it just in case :)

        # finish the session!
        self.finish_session()


    def finish_session(self):
        print("\n=== Session Finished ===")
        self.stop_evt.set()  # gracefully shut vision thread
        self.vision_thr.join()
        self.generate_pdf_report()  # create the PDF
        
    # needed for determining the time length of the trial, and maybe eventually, the prompting time/delay (zero-second, two-second, independent)
    def get_time(self):
        elapsed_time = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        return f"{minutes:02}:{seconds:02}"
        
    # If you need to change Furhat's LED:
    def set_led(self, color):
        if color == "red":
            self.furhat.set_led(red=255, green=0, blue=0)
        elif color == "blue":
            self.furhat.set_led(red=50, green=50, blue=200)
        elif color == "green":
            self.furhat.set_led(red=50, green=200, blue=50)
        else:
            self.furhat.set_led(red=0, green=0, blue=0)

    # The second and main LLM call, to determine the interventionist's accuracy
    def evaluate_technician_action(self, trial_number, bt_instruction,
                                   bt_cs, child_response, vision_action, llm):

        prompted_time = self.get_time()

        prompt = f"""
        You are an expert in evaluating Applied Behavior Analysis (ABA) Discrete Trial Training (DTT).

        A dictionary containing the trial type, explanation, and the BT's ideal instruction / reinforcement / error‑correction is given here: {self.trials}
        For imitation trials, you will see "Do this [...]". Please know that the "..." is not meant to be said, but rather the gesture.
        It is OK if the child's name is used.
        If the phrasing is nearly fully correct, mark it as correct. If it is moderately wrong, mark it as "Acceptable" and briefly explain why.
        If it is noticeably wrong, mark it as "Incorrect" and explain why.
        The BT said the instruction (SD): {bt_instruction}
        The gesture detected by vision is: {vision_action}
        The child's response: {child_response}
        The BT's consequence strategy (CS): {bt_cs}
        The inferred trial number is {trial_number}. Current time: {prompted_time}

        1) Provide the trial type.
        2) Echo the BT's instruction and if they had a gesture.
        3) Determine if the evaluation is "Correct", "Acceptable", or "Incorrect"
        4) If acceptable or incorrect, explain; else "N/A".
        5) Echo the child's response.
        6) Echo the BT's CS.
        7) Determine if the CS is a reinforcement or error‑correction (exclude prompts for now)
        8) Evaluate if the CS is correct for the child's response.
        9) If incorrect, explain; else "N/A".
        10) Provide the timestamp.

        Respond ONLY with valid JSON matching:

        {{
            "Trial Type": "...",
            "Received Instruction": "'...'",
            "Instruction Evaluation": "...",
            "Instruction Feedback": "...",
            "Child Response": "...",
            "CS Received": "'...'",
            "CS Category": "...",
            "CS Evaluation": "...",
            "CS Feedback": "...",
            "Time": "..."
        }}
        """

        try:
            response      = llm.invoke([{"role": "system", "content": prompt}])
            response_text = response.content if hasattr(response, "content") else str(response)
            print("Evaluation:", response_text)

            json_start = response_text.find("{")
            json_end   = response_text.rfind("}") + 1
            if json_start == -1 or json_end == -1:
                raise ValueError("Response not JSON")

            result_json = json.loads(response_text[json_start:json_end])

        except Exception as e:
            print("Evaluation error:", e)
            result_json = {
                "Trial Type": "error",
                "Received Instruction": bt_instruction,
                "Instruction Evaluation": "error",
                "Instruction Feedback": str(e),
                "Child Response": child_response,
                "CS Received": bt_cs,
                "CS Category": "error",
                "CS Evaluation": "error",
                "CS Feedback": "error",
                "Time": prompted_time
            }

        return result_json
                                       
    # PDF generation
    def generate_pdf_report(self):
        styles = getSampleStyleSheet()
        normalStyle = styles['BodyText']; normalStyle.fontSize = 8

        expected_columns = [
            "Trial Type", "Received Instruction", "Instruction Evaluation",
            "Instruction Feedback", "Child Response", "CS Received",
            "CS Category", "CS Evaluation", "CS Feedback", "Time"
        ]

        header = [Paragraph(col, normalStyle) for col in expected_columns]
        table_data = [header]

        for result in self.evaluation_results:
            row = [Paragraph(str(result.get(col, "")), normalStyle)
                   for col in expected_columns]
            table_data.append(row)

        pdf_file = "Evaluation_Report.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=landscape(letter),
                                leftMargin=0.5*inch, rightMargin=0.5*inch,
                                topMargin=0.5*inch,  bottomMargin=0.5*inch)

        col_w = doc.width / len(expected_columns)
        table = Table(table_data, colWidths=[col_w]*len(expected_columns))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.gray),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        doc.build([table])
        print(f"PDF Report generated: {pdf_file}")
        os.startfile(pdf_file)
        

# NAO behavior function
def naoBehavior(name: str):
    url = f"http://141.210.88.206:5000/behavior?name={name.replace(' ', '%20')}"
    print(url)
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error: behavior {name} failed to run\n{r.content}")
        

def main():
    session = DTTSession()
    session.run_session()


if __name__ == "__main__":
    main()

    
