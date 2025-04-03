from langchain_openai import ChatOpenAI
import json
import time
import LookupTables
from furhat_remote_api import FurhatRemoteAPI


class DTTSession:
    def __init__(self):
        # Connect furhat
        self.furhat = FurhatRemoteAPI("localhost")

        # Load lookup tables
        self.prompts = LookupTables.sds
        self.states = LookupTables.states
        self.child_responses = LookupTables.child_correct_responses
        self.reinforcements = LookupTables.reinforcements

        # Initialize time and starting state
        self.start_time = time.time()
        self.DTT_state = "WELCOME"

        # Initialize LLMs (only one will be used when deployed)
        self.gpt4o_mini = self._initialize_gpt4o_mini()
        self.gpt4o = self._initialize_gpt4o()
        self.gpto3_mini = self._initialize_gpto3_mini()

        # Store user inputs per trial (if needed)
        self.trial_responses = {}

    def _initialize_gpt4o_mini(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_retries=2,
            api_key="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"  # Hide or store for safety
        )

    def _initialize_gpt4o(self):
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=2,
            api_key="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"
        )

    def _initialize_gpto3_mini(self):
        return ChatOpenAI(
            model="gpt-o3-mini",
            temperature=0,
            max_retries=2,
            api_key="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"
        )

    def run_session(self):

     # Loop through each of x-number of trials (for now, 5). For each trial:
        # 1. Set LED color to indicate readiness
        # 2. Listen & chunk input (or get a manual test phrase)
        # 3. Evaluate input using evaluate_technician_action
        # 4. Possibly store or display results

        for trial_num in range(1, 6):
            print(f"\n=== Trial {trial_num} ===")

            # LED: indicate user can talk (blue)
            self.set_led("blue")

            # Get input from the user
            # For demonstration, I'm using a manual function below
            bt_prompt = self.get_manual_prompt(trial_num)
            bt_reinforcement = self.get_manual_reinforcement(trial_num)

            # LED: indicate we are processing input (red)
            self.set_led("red")

            # Evaluate action
            result = self.evaluate_technician_action(bt_prompt, bt_reinforcement, self.DTT_state, self.gpt4o)

            # Update the session state if we have a new state
            if "State" in result:
                self.DTT_state = result["State"]

            # Save the result
            self.trial_responses[trial_num] = result

        self.finish_session()

    def finish_session(self):
        # Example: print all results
        print("\n=== Session Finished. Trial Evaluations ===")
        for tnum, evaluation in self.trial_responses.items():
            print(f"Trial {tnum}: {evaluation}")

    def get_manual_prompt(self, trial_num):
        test_prompts = {
            1: "Emily, what do you want to work for?",
            2: "Do this",
            3: "Wave",
            4: "What is this?",
            5: "How do I feel?",
            # ...
        }
        return test_prompts.get(trial_num)

    def get_manual_reinforcement(self, trial_num):
        test_reinforcements = {
            1: "You want to work for stickers? Ok!",
            2: "Good job!",
            3: "Good job!",
            4: "No, this is a car.",
            5: "Not quite! This is a happy face.",
            # ...
        }
        return test_reinforcements.get(trial_num)

    def get_time(self):
        elapsed_time = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        return f"{minutes:02}:{seconds:02}"

    def set_led(self, color):
        if color == "red":
            self.furhat.set_led(red=200, green=50, blue=50)
        elif color == "blue":
            self.furhat.set_led(red=50, green=50, blue=200)
        elif color == "green":
            self.furhat.set_led(red=50, green=200, blue=50)
        else:
            self.furhat.set_led(red=0, green=0, blue=0)

        # time.sleep(1)

    def evaluate_technician_action(self, bt_prompt, bt_reinforcement, previous_state, llm):
        prompted_time = self.get_time()

        prompt = f"""
        You are an expert in evaluating Applied Behavior Analysis (ABA) Discrete Trial Training (DTT) for Behavior Technicians.

        Each trial for this study is briefly explained in the {self.states} dictionary.
        There are three categories of trials: zero-second prompting, two-second prompting, and independent responding.
        Each category has five types of trials: Manding/motivation, imitation, reception, tact/labeling, and emotions.
        Each action and state with the same name align. 

        Each trial has an ideal phrasing the BT should use, outlined in {self.prompts} for prompts and {self.reinforcements} for reinforcements.
        However, it is OK if the BT phrasing is slightly off of the actions dictionary.
        It is also OK if the BT uses the child's name (the child's name will always be Emily).

        The previous DTT state was: {previous_state}.
        The BT just said the following prompt: {bt_prompt}.
        The BT just said the following reinforcement: {bt_reinforcement}.
        The current time is {prompted_time}.

        1) First, determine the **correct DTT state** based on this action by referring to {self.states}.
        2) Then, evaluate if the BT's given prompt is **correct** for that state by referring to {self.prompts}.
        3) If incorrect, explain the mistake and what should have been done instead. If correct, do not print this field.
        4) Then, evaluate if the BT's given reinforcement is **correct** for the child's response to the prompt by referring to {self.child_responses} and {self.reinforcements}
        5) If incorrect, explain the mistake and what should have been done instead. If correct, do not print this field.
        6) Track the prompted time.
        
        Respond ONLY in valid JSON format (no extra text). Output should strictly follow this format:
        
        {{
            "State": "<updated_state>",
            "Prompt Evaluation": "<correct/incorrect>",
            "Prompt Feedback": "<detailed explanation>",
            "Reinforcement Evaluation": "<correct/incorrect>",
            "Reinforcement Feedback": "<detailed explanation>",
            "Time": "<timestamp>"
        }}
    

        """

        try:
            response = llm.invoke([{"role": "system", "content": prompt}])
            response_text = response.content if hasattr(response, "content") else str(response)

            print("Evaluation:", response_text)  # Debugging

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


def main():
    # Create a session
    session = DTTSession()

    # Optionally do some quick debug tests *before* or *after* session
    # session.debug_tests()

    # Run the full 10-trial session
    session.run_session()


if __name__ == "__main__":
    main()
