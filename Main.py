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

class DTTSession:
    def __init__(self):
        # -------- robots & LLM --------------------------------------------
        self.furhat = FurhatRemoteAPI("141.210.88.11")
        self.furhat.attend(user="CLOSEST")
        time.sleep(2)

        self.trials   = LookupTables.trials
        self.start_time = time.time()
        self.DTT_state  = "WELCOME"
        self.evaluation_results = []

        self.gpt41 = self._initialize_gpt41()

        # -------- vision thread -------------------------------------------
        self.vision_q   = queue.Queue()
        self.stop_evt   = threading.Event()
        self.vision_thr = VisionDetector(self.vision_q, self.stop_evt)
        self.vision_thr.start()


    def _initialize_gpt41(self):
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            max_retries=2,
            api_key=""
        )

    def run_session(self):

        time.sleep(3)
        # so I can run to the other side of the table after pressing play, lol

        for trial_iteration in range(1, 4):
            print(f"\n=== Trial {trial_iteration} ===")

            # self.set_led("red")

            if trial_iteration == 1:
                self.furhat.say(text="Please begin now!")
                time.sleep(2) 

            #else:
                #self.furhat.say(text="Please provide your next instruction.")

            # ======== INSTRUCTION (SD) ====================================
            # clear any stale gestures so we only catch those made _now_
            while not self.vision_q.empty():
                _ = self.vision_q.get_nowait()

            # self.set_led("blue")
            naoBehavior(".lastUploadedChoregrapheBehavior/LightGreen")
            instruction_result = self.furhat.listen()
            time.sleep(1)
            # self.set_led("red")

            if hasattr(instruction_result, "message"):
                bt_instruction = instruction_result.message
            elif isinstance(instruction_result, dict):
                bt_instruction = instruction_result.get("message", "")
            else:
                bt_instruction = str(instruction_result)

            # pull the first gesture detected during this SD
            vision_action = "none"
            while not self.vision_q.empty():
                vision_action = self.vision_q.get_nowait()

            # -------- LLM: classify trial ---------------------------------
            prompt = f"""
            You are an ABA DTT expert. A behavior technician is being trained to perform DTT with a child.
            Here is what the BT has said to the child as an instruction (Discriminative Stimulus, or SD): {bt_instruction}
            While speaking, the BT’s gesture‑tracking camera detected: {vision_action}
            If an imitation instruction is presented, but there are is no gesture detected, please return "0"
            Here is a dictionary of all of the possible and ideal phrasing for the instructions: {self.trials}

            Your job is to scan the dictionary of possible trials and tell me which number trial the instruction BEST corresponds to.
            It's OK if the phrasing is slightly off and it is also OK if the child's name (Emily) is used.
            However, if the instruction absolutely does not match any of the trials, return "0".

            Respond with ONLY the trial number (e.g., "5"). No other words.
            """

            trial_number = "N/A"
            try:
                response      = self.gpt41.invoke([{"role": "system", "content": prompt}])
                trial_number  = response.content if hasattr(response, "content") else str(response)
                print("LLM Output:", trial_number)
            except Exception as e:
                print("Error while processing instruction:", e)

            # ======== CHILD REACTION (50/50) ===============================
            reaction = random.randrange(1, 3)
            #reaction = 1

            child_response = "N/A"
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
                    child_response = "N/A"
                case _:
                    # NOT A VALID OUTPUT FROM LLM
                    child_response = "N/A 2"     

            # ======== CONSEQUENCE STRATEGY (BT) ============================
            time.sleep(2.5)
            # self.set_led("blue")
            naoBehavior(".lastUploadedChoregrapheBehavior/LightGreen")
            cs_result = self.furhat.listen()
            time.sleep(1)
            # self.set_led("red")

            if hasattr(cs_result, "message"):
                bt_cs = cs_result.message
            elif isinstance(cs_result, dict):
                bt_cs = cs_result.get("message", "")
            else:
                bt_cs = str(cs_result)

            # ======== LLM: evaluate full trial ============================
            result = self.evaluate_technician_action(
                trial_number,
                bt_instruction,
                bt_cs,
                child_response,
                vision_action,
                self.gpt41
            )

            self.evaluation_results.append(result)

            if "Trial Type" in result:
                self.DTT_state = result["Trial Type"]

        self.finish_session()


    def finish_session(self):
        print("\n=== Session Finished ===")
        self.stop_evt.set()          # gracefully shut vision thread
        self.vision_thr.join()
        self.generate_pdf_report()
        

    def get_time(self):
        elapsed_time = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        return f"{minutes:02}:{seconds:02}"
        

    def set_led(self, color):
        if color == "red":
            self.furhat.set_led(red=255, green=0, blue=0)
        elif color == "blue":
            self.furhat.set_led(red=50, green=50, blue=200)
        elif color == "green":
            self.furhat.set_led(red=50, green=200, blue=50)
        else:
            self.furhat.set_led(red=0, green=0, blue=0)


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

    
