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
        self.trials_list = LookupTables.trials_list
        self.actions = LookupTables.actions
        self.states = LookupTables.states
        self.incorrect_actions = LookupTables.incorrect_actions

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
            api_key="YOUR_API_KEY"  # Hide or store safely
        )

    def _initialize_gpt4o(self):
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=2,
            api_key="YOUR_API_KEY"
        )

    def _initialize_gpto3_mini(self):
        return ChatOpenAI(
            model="gpt-o3-mini",
            temperature=0,
            max_retries=2,
            api_key="YOUR_API_KEY"
        )

    def run_session(self):
        """
        Loop through 10 trials. For each trial:
          1. Set LED color to indicate readiness
          2. Listen & chunk input (or get a manual test phrase)
          3. Evaluate input using evaluate_technician_action
          4. Possibly store or display results
        """
        for trial_num in range(1, 11):
            print(f"\n=== Starting Trial {trial_num} ===")

            # LED: indicate user can talk (blue)
            self.set_led("blue")

            # Get input from the user or do something manual
            # For demonstration, I'm using a manual function below
            user_input = self.debug_get_manual_input(trial_num)

            # LED: indicate we are processing input (red)
            self.set_led("red")

            # Evaluate action
            result = self.evaluate_technician_action(user_input, self.DTT_state, self.gpt4o_mini)

            # Update the session state if we have a new state
            if "State" in result:
                self.DTT_state = result["State"]

            # Save the result
            self.trial_responses[trial_num] = result

        self.finish_session()

    def finish_session(self):
        """
        Optionally do some wrap-up or final summary of results.
        """
        # Example: print all results
        print("\n=== Session Finished. Trial Evaluations ===")
        for tnum, evaluation in self.trial_responses.items():
            print(f"Trial {tnum}: {evaluation}")

    def debug_get_manual_input(self, trial_num):
        """
        For manual testing, you might input from console or
        hard-code some strings for each trial.
        Real scenario: you'd use furhat.listen() or a chunking approach.
        """
        # In a real scenario, you'd do something like:
        # response_text = self.furhat.listen()
        # Or a chunk-based approach with repeated calls to self.furhat.listen()
        # until you detect a 'pause' or 'end of chunk.'

        # For now, returning a placeholder
        test_inputs = {
            1: "What do you want to work for?",
            2: "Emily, what do you want to work for?",
            3: "What would you like to work for?",
            4: "Let's try something else.",
            # ...
        }
        return test_inputs.get(trial_num, "Default test input")

    def get_time(self):
        """
        Calculate elapsed time from the start of the session.
        """
        elapsed_time = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        return f"{minutes:02}:{seconds:02}"

    def set_led(self, color):
        """
        LED feedback - example values only, adjust as needed.
        """
        # Adjust color mapping as you like:
        if color == "red":
            self.furhat.set_led(red=200, green=50, blue=50)
        elif color == "blue":
            self.furhat.set_led(red=50, green=50, blue=200)
        elif color == "green":
            self.furhat.set_led(red=50, green=200, blue=50)
        else:
            self.furhat.set_led(red=0, green=0, blue=0)

        # Time delay if you want to hold color for a bit:
        # time.sleep(1)

    def evaluate_technician_action(self, technician_action, previous_state, llm):
        """
        Evaluate the technician action using the LLM, returning JSON data.
        """
        prompted_time = self.get_time()

        prompt = f"""
        You are an expert in evaluating Applied Behavior Analysis (ABA) Discrete Trial Training (DTT) for Behavior Technicians.

        Each trial for this study is briefly explained in the {self.states} dictionary.
        There are three categories of trials: zero-second prompting, two-second prompting, and independent responding.
        Each category has five types of trials: Manding/motivation, imitation, reception, tact/labeling, and emotions.
        Each action and state with the same name align. 

        Each trial has an ideal phrasing the BT should use, outlined in {self.actions}.
        However, it is OK if the BT phrasing is slightly off of the actions dictionary.
        You can reference incorrect responses outlined in {self.incorrect_actions}.
        It is also OK if the BT uses the child's name (the child's name will always be Emily).

        The previous DTT state was: {previous_state}.
        The technician just performed the following action: {technician_action}.
        The current time is {prompted_time}.

        1) First, determine the **correct DTT state** based on this action by referring to {self.states}.
        2) Then, evaluate if the action is **correct** for that state by referring to {self.actions}.
        3) If incorrect, explain the mistake and what should have been done instead.
        4) Keep track of the prompted time.

        Respond ONLY in valid JSON format(no extra text, no explanation). Output should strictly follow this format:
        ```json
        {{
            "State": "<updated_state>",
            "BT Evaluation": "<correct/incorrect>",
            "Feedback": "<detailed explanation>",
            "Time": "<timestamp>"
        }}
        ```

        There may be **multiple trials** for each action. If that's the case, identify each trial, and make **separate** json entries with the above format.
        The objective is to show the BT your evaluation at the end of the training so that they can improve.
        """

        try:
            response = llm.invoke([{"role": "system", "content": prompt}])
            response_text = response.content if hasattr(response, "content") else str(response)

            print("LLM Response (raw):", response_text)  # Debugging

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