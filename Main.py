import sys
import json
import time
import random
import requests
from langchain_openai import ChatOpenAI
from furhat_remote_api import FurhatRemoteAPI
import LookupTables
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

class DTTSession:
    def __init__(self):
        # Connect Furhat
        self.furhat = FurhatRemoteAPI("141.210.88.11")
        self.furhat.say(text="Connected successfully")
        self.furhat.attend(user="CLOSEST")
        naoBehavior(".lastUploadedChoregrapheBehavior/Connected")

        time.sleep(2)

        # Load lookup tables
        self.instructions = LookupTables.sds  # formerly self.prompts
        self.states = LookupTables.states
        self.child_responses = LookupTables.child_correct_responses
        self.reinforcements = LookupTables.reinforcements

        self.start_time = time.time()
        self.DTT_state = "WELCOME"

        # To collect evaluation results from each trial
        self.evaluation_results = []

        self.gpt4o = self._initialize_gpt4o()

    def _initialize_gpt4o(self):
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=2,
            api_key="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"
        )

    def run_session(self):
        """
        Loop through each trial. For each trial:
          1. Ensure LED is red by default (BT should not speak).
          2. Provide the BT with an instruction, then set LED to blue before listening.
          3. After listening, revert LED to red.
          4. Repeat for the reinforcement.
          5. Evaluate input using evaluate_technician_action.
          6. Store the trial results.
        """
        for trial_num in range(1, 2):
            print(f"\n=== Trial {trial_num} ===")

            self.set_led("red")

            if (trial_num == 1):
                self.furhat.say(text = "Please begin now!")
            else:
                self.furhat.say(text="Please provide your next instruction.")

            time.sleep(2.5)
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

            num = random.randrange(1, 3, 1)  # random num (1 or 2)
            child_response = "N/A"

            if trial_num == 1:
                naoBehavior(".lastUploadedChoregrapheBehavior/behavior_1")
                child_response = "I want stickers!" 
            elif trial_num == 2:
                if num == 1:
                    naoBehavior(".lastUploadedChoregrapheBehavior/TouchHead")
                    child_response = "[touches head]"
                elif num == 2:
                    naoBehavior(".lastUploadedChoregrapheBehavior/DontTouchHead")
                    child_response = "[doesn't touch head]"
            elif trial_num == 3:
                if num == 1:
                    naoBehavior(".lastUploadedChoregrapheBehavior/ClapHAnds")
                    child_response = "[claps hands]"
                elif num == 2:
                    naoBehavior(".lastUploadedChoregrapheBehavior/DontClapHands")
                    child_response = "[doesn't clap hands]"

            time.sleep(2.5)
            self.set_led("blue")
            reinforcement_result = self.furhat.listen()
            time.sleep(1)
            self.set_led("red")

            if hasattr(reinforcement_result, "message"):
                bt_reinforcement = reinforcement_result.message
            elif isinstance(reinforcement_result, dict):
                bt_reinforcement = reinforcement_result.get("message", "")
            else:
                bt_reinforcement = str(reinforcement_result)

            # Evaluate technician action (processing phase, LED remains red)
            result = self.evaluate_technician_action(
                bt_instruction,
                bt_reinforcement,
                child_response,
                self.DTT_state,
                self.gpt4o
            )

            # Store the evaluation result for later reporting
            self.evaluation_results.append(result)

            # Update the session state if we have a new state
            if "State" in result:
                self.DTT_state = result["State"]

        self.finish_session()

    def finish_session(self):
        print("\n=== Session Finished ===")
        # Generate PDF report based on evaluation results
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

    def evaluate_technician_action(self, bt_instruction, bt_reinforcement, child_response, previous_state, llm):
        prompted_time = self.get_time()

        prompt = f"""
        You are an expert in evaluating Applied Behavior Analysis (ABA) Discrete Trial Training (DTT) for Behavior Technicians.

        Each trial for this study is briefly explained in the {self.states} dictionary.
        There are three categories of trials: zero-second prompting, two-second prompting, and independent responding.
        Each category has five types of trials: Manding/motivation, imitation, reception, tact/labeling, and emotions.
        Each action and state with the same name align.

        Each trial has an ideal phrasing the BT should use, outlined in {self.instructions} for instructions and {self.reinforcements} for reinforcements.
        However, it is OK if the BT phrasing is slightly off of the actions dictionary.
        It is also OK if the BT uses the child's name (the child's name will always be Emily).

        The previous DTT state was: {previous_state}.
        The BT just said the following instruction: {bt_instruction}.
        The BT just said the following consequence strategy (CS): {bt_reinforcement}.
        The child's response to the BT's instruction is: {child_response}
        The current time is {prompted_time}.

        1) First, determine the **correct DTT state** based on this action by referring to {self.states}.
        2) Provide the BT's instruction.
        3) Then, evaluate if the BT's given instruction is **correct** for that state by referring to {self.instructions}.
        4) If incorrect, explain the mistake and what should have been done instead. If correct, do not print this field.
        5) Provide the child's response.
        6) Provide the BT's consequence strategy (CS).
        7) Determine if the BT's CS is an error-correction, prompt, or reinforcement.
        8) Then, evaluate if the BT's given CS is **correct** for the child's response to the instruction by referring to {self.child_responses} and {self.reinforcements}.
        9) If incorrect, explain the mistake and what should have been done instead. If correct, do not print this field.
        10) Provide the prompted time.
        
        Respond ONLY in valid JSON format (no extra text). Output should strictly follow this format:
        
        {{
            "State": "<updated_state>",
            "Received Instruction:" : "<what the BT said>",
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
                "State": previous_state,
                "BT Evaluation": "error",
                "Feedback": f"Error in processing response. Raw output: {response_text}",
                "Time": prompted_time
            }
        except Exception as e:
            print("Error during LLM processing:", e)
            result_json = {
                "State": previous_state,
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
        """
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.units import inch

        # Define the expected columns in order; if some keys are missing in a row, we'll leave them blank.
        expected_columns = [
            "State",
            "Received Instruction:",
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
        table_data = [expected_columns]
        for result in self.evaluation_results:
            # Create a row for each evaluation, using empty strings for missing keys
            row = [result.get(col, "") for col in expected_columns]
            table_data.append(row)
        
        # Define the PDF file name and document in landscape orientation
        pdf_file = "Evaluation_Report.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=landscape(letter),
                                leftMargin=0.5*inch, rightMargin=0.5*inch,
                                topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Calculate the available width for the table
        available_width = doc.width
        # Divide the available width equally among all columns
        num_cols = len(expected_columns)
        col_widths = [available_width / num_cols] * num_cols
        
        # Create a table with the computed column widths
        table = Table(table_data, colWidths=col_widths)
        
        # Add style to the table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Reduce font size to help fit data
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)
        
        # Build the PDF
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
