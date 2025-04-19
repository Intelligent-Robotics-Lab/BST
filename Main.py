import sys
import json
import time
import random
import requests
import LookupTables

from langchain_openai import ChatOpenAI
from furhat_remote_api import FurhatRemoteAPI
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

introduction = '''

Hello! Welcome to behavior technician training. I'm Furhat, and I'll be your training partner today. 

We're going to practice Discrete Trial Training, or DTT, together. I'll play the role of the supervisor, and your goal is to teach Emily the tasks in front of you!

During our session, I'll be listening closely to your instructions, watching your movements, and evaluating how you are interacting with Emily. Remember to speak clearly, provide positive reinforcement, and follow the DTT protocol step-by-step.

Let me demonstrate.

'''

class DTTSession:
    def __init__(self):
        # Connect Furhat
        self.furhat = FurhatRemoteAPI("141.210.88.11")
        #self.furhat.say(text="Connected successfully")
        self.furhat.attend(user="CLOSEST")
        #naoBehavior(".lastUploadedChoregrapheBehavior/Connected")

        time.sleep(2)

        # Load lookup tables
        self.trials = LookupTables.trials

        self.start_time = time.time()
        self.DTT_state = "WELCOME"

        # To collect evaluation results from each trial
        self.evaluation_results = []

        self.gpt41 = self._initialize_gpt41()

    def _initialize_gpt41(self):
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            max_retries=2,
            api_key="", # Put your API key here
        )
    
    def introduction(self):
        self.furhat.say(text=introduction)
        time.sleep(34)

    def run_session(self):

        # self.introduction()

        for trial_iteration in range(1, 2):
            print(f"\n=== Trial {trial_iteration} ===")

            self.set_led("red")

            if (trial_iteration == 1):
                self.furhat.say(text = "Please begin now!")
            else:
                self.furhat.say(text="Please provide your next instruction.")

            # # # # # # INSTRUCTION (DISCRIMINATIVE STIMULUS) # # # # # #

            time.sleep(2)
            self.set_led("blue")
            instruction_result = self.furhat.listen()
            time.sleep(1)
            self.set_led("red")

            if hasattr(instruction_result, "message"):
                bt_instruction = instruction_result.message
            elif isinstance(instruction_result, dict):
                bt_instruction = instruction_result.get("message", "")
            else:
                bt_instruction = str(instruction_result)

            prompt = f"""
            You are an ABA DTT expert. A behavior technician is being trained to perform DTT with a child.
            Here is what the BT has said to the child as an instruction (Discriminitive Stimulus, or SD): {bt_instruction} 
            Here is a dictionary of all of the possible and ideal phrasing for the instructions: {self.trials}
            Your job is to scan the dictionary of possible trials and tell me which number trial the instruction BEST corresponds to.
            It's OK if the phrasing is slightly off. You should be very lenient, also referring to the "explanation" to double check what the BT may have meant.
            It is also OK if the child's name is used (Emily).
            However, if the instruction absolutely does not match any of the trials, please return "0".
            You should ONLY reply with the one number of the corresponding trial. Do not add any other words or numbers.
            For example, your output may be "5" or "12" or "0" etc. Do not provide any extra text.
            """

            trial_number = "N/A"

            try:
                response = self.gpt41.invoke([{"role": "system", "content": prompt}])
                trial_number = response.content if hasattr(response, "content") else str(response)

                print("LLM Output: ", trial_number)
            except Exception as e:
                print("Error while processing interventionist's instruction in the LLM:", e)


            # # # # # # CHILD REACTION # # # # # #

            reaction = random.randrange(1, 3, 1)  # randomly determine if child responds correctly or incorrectly (50%/50%)
            # Which trial has the BT initiated with their instruction? Refer to "trials" lookup table
            
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
                        naoBehavior("bststudy/Clap Prompt")
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
                        naoBehavior("bststudy/Clap Hands")
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


            # # # # # # CONSEQUENCE STRATEGY # # # # # #

            time.sleep(2.5)
            self.set_led("blue")
            cs_result = self.furhat.listen()
            time.sleep(1)
            self.set_led("red")

            if hasattr(cs_result, "message"):
                bt_cs = cs_result.message
            elif isinstance(cs_result, dict):
                bt_cs = cs_result.get("message", "")
            else:
                bt_cs = str(cs_result)

            # Evaluate technician action
            result = self.evaluate_technician_action(
                trial_number,
                bt_instruction,
                bt_cs,
                child_response,
                self.gpt41
            )

            # Store the evaluation result for PDF at the end
            self.evaluation_results.append(result)

            # Update the session state if we have a new state
            if "Trial Type" in result:
                self.DTT_state = result["Trial Type"]
                
        self.finish_session()

    def finish_session(self):
        print("\n=== Session Finished ===")
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

    def evaluate_technician_action(self, trial_number, bt_instruction, bt_cs, child_response, llm):
        prompted_time = self.get_time()

        prompt = f"""
        You are an expert in evaluating Applied Behavior Analysis (ABA) Discrete Trial Training (DTT) for Behavior Technicians.

        A dictionary containing the type of trial, explanation, and the BT's ideal instruction/reinforcement/error corrections are listed in the following "trials" dictionary: {self.trials}
        The five types of trials we use are: Manding/motivation, imitation, reception, tact/labeling, and emotions.

        As you know, the BT begins by reciting an instruction, also known as a Discriminative Stimulus (or SD).
        The child then responds correctly or incorrectly.
        Then, the BT follows up with either a reinforcement (if the child responded correctly) OR an error-correction (if the child responded incorrectly).
        This is called the BT's consequence strategy (or CS).

        You are to refer to the provided "trials" dictionary for determining how well the BT's instructions and consequence strategies align with the ideal verbals.
        However, it is OK if the BT phrasing, grammar, and word choice is off, as long as it generally retains its meaning. 
        It is also OK if the BT uses the child's name (the child's name will always be Emily).

        The BT said the following instruction: {bt_instruction}.
        The child's response to the BT's instruction is: {child_response}
        The BT's consequence strategy (CS) is: {bt_cs}.
        The current time is {prompted_time}.
        The trial number is: {trial_number}
        You can find the trial type from the "type" element of the "trials" dictionary.

        1) Provide the trial type
        2) Provide the BT's instruction.
        3) Then, evaluate if the BT's given instruction is **correct** for that state by referring to the available instructions.
        4) If incorrect, explain the mistake and what should have been done instead. If correct, print N/A.
        5) Provide the child's response.
        6) Provide the BT's consequence strategy (CS).
        7) Determine if the BT's CS is an error-correction, prompt, or reinforcement.
        8) Then, evaluate if the BT's given CS is **correct** given the correct response type.
        9) If incorrect, explain the mistake and what should have been done instead. If correct, print N/A.
        10) Provide the prompted time.
        
        Respond ONLY in valid JSON format (no extra text). Output should strictly follow this format:
        
        {{
            "Trial Type": "<trial type>",
            "Received Instruction" : "<what the BT said>",
            "Instruction Evaluation": "<correct/incorrect>",
            "Instruction Feedback": "<detailed explanation>",
            "Child Response": "<child's response>",
            "CS Received": "<what the BT said>",
            "CS Category": "<the BT's chosen consequence strategy>",
            "CS Evaluation": "<correct/incorrect>",
            "CS Feedback": "<detailed explanation>",
            "Time": "<timestamp>"
        }}
        """

        try:
            response = llm.invoke([{"role": "system", "content": prompt}])
            response_text = response.content if hasattr(response, "content") else str(response)

            print("Evaluation:", response_text)  # Debug/logging

            # Extract JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start == -1 or json_end == -1:
                raise ValueError("Response is not properly formatted as JSON")

            json_response = response_text[json_start:json_end]
            result_json = json.loads(json_response)

        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            print("Raw Response:", response_text)
            result_json = {
                "State": "error",
                "BT Evaluation": "error",
                "Feedback": f"Error in processing response. Raw output: {response_text}",
                "Time": prompted_time
            }
        except Exception as e:
            print("Error during LLM processing:", e)
            result_json = {
                "State": "error",
                "BT Evaluation": "error",
                "Feedback": "Unexpected error occurred.",
                "Time": prompted_time
            }

        return result_json

    def generate_pdf_report(self):
        """
        Converts the evaluation results into a table and exports it to a PDF.
        Each row corresponds to one JSON evaluation result.
        Each column represents one key-value pair from the JSON (one line).
        Text in each cell is wrapped using Paragraph objects.
        """

        # Create a paragraph style that we will use for each cell
        styles = getSampleStyleSheet()
        normalStyle = styles['BodyText']
        normalStyle.fontSize = 8

        expected_columns = [
            "Trial Type",
            "Received Instruction",
            "Instruction Evaluation",
            "Instruction Feedback",
            "Child Response",
            "CS Received",
            "CS Category",
            "CS Evaluation",
            "CS Feedback",
            "Time"
        ]
        
        # Create the table data (first row as header)
        # Wrap header texts in Paragraph objects as well.
        header = [Paragraph(col, normalStyle) for col in expected_columns]
        table_data = [header]
        
        for result in self.evaluation_results:
            row = []
            for col in expected_columns:
                cell_text = str(result.get(col, ""))
                p = Paragraph(cell_text, normalStyle)
                row.append(p)
            table_data.append(row)
        
        pdf_file = "Evaluation_Report.pdf"
        doc = SimpleDocTemplate(
            pdf_file, 
            pagesize=landscape(letter),
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch
        )
        
        available_width = doc.width
        num_cols = len(expected_columns)
        col_widths = [available_width / num_cols] * num_cols
        
        table = Table(table_data, colWidths=col_widths)
        
        # Add style to the table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),  
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)
        
        elements = [table]
        doc.build(elements)
        print(f"PDF Report generated: {pdf_file}")


def main():
    # Create session
    session = DTTSession()

    # Run session
    session.run_session()

def naoBehavior(name: str):
    url = f"http://141.210.88.206:5000/behavior?name={name.replace(' ', '%20')}"
    print(url)
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error: behavior {name} failed to run")
        print(str(r.content))

if __name__ == "__main__":
    main()
