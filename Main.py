from langchain_openai import ChatOpenAI
import json
import time
import LookupTables
from furhat_remote_api import FurhatRemoteAPI

furhat = FurhatRemoteAPI("localhost")
current_state = "WELCOME"
start_time = 0

llm = ChatOpenAI(
    model="gpt-4o-mini", # replacement of 3.5-turbo
    temperature=0,
    max_retries=2,
    api_key="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"
)

def start():
    furhat.say(text="Welcome to training. Begin now.")

    global start_time, current_state
    start_time = time.time()

    # result = furhat.listen()

    # Test responses.
    response = evaluate_technician_action("What do you want to work for?", "WELCOME")
    response = evaluate_technician_action("Emily, what do you want to work for?", "WELCOME")
    response = evaluate_technician_action("What would you like to work for?", "WELCOME")

    if "State" in response:
        current_state = response["State"]

    furhat.say(text="I've evaluated your first action. Your new state is: " + current_state)


def get_time():
  elapsed_time = time.time() - start_time
  minutes, seconds = divmod(int(elapsed_time), 60)
  formatted_time = f"{minutes:02}:{seconds:02}"

  return formatted_time


# LED will be turned on whenever it is detecting user voice
def led():
    furhat.set_led(red=200, green=50, blue=50)
    time.sleep(10)


def evaluate_technician_action(technician_action, previous_state):
    prompted_time = get_time()
    actions = LookupTables.actions
    states = LookupTables.states
    incorrect_actions = LookupTables.incorrect_actions

    prompt = f"""
    You are an expert in evaluating Applied Behavior Analysis (ABA) Discrete Trial Training (DTT) for Behavior Technicians.
    
    Each trial for this study is briefly explained in the {states} dictionary.
    There are three categories of trials: zero-second prompting, two-second prompting, and independent responding.
    Each category has five types of trials: Manding/motivation, imitation, reception, tact/labeling, and emotions.
    Each action and state with the same name align. 
        
    Each trial has an ideal phrasing the BT should use, outlined in {actions}.
    However, it is OK if the BT phrasing is slightly off of the actions dictionary. 
    You can reference incorrect responses outlined in {incorrect_actions}.
    It is also OK if the BT uses the child's name (the child's name will always be Emily).
    
    The previous DTT state was: {previous_state}.
    The technician just performed the following action: {technician_action}.
    The current time is {prompted_time}.
    
    1) First, determine the **correct DTT state** based on this action by referring to {states}.
    2) Then, evaluate if the action is **correct** for that state by referring to {actions}.
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

        # Debugging print:
        print(response_text)

        # Ensure we extract only the JSON portion
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("Response is not properly formatted as JSON")

        json_response = response_text[json_start:json_end]

        # Parse JSON response
        result_json = json.loads(json_response)

        #print(result_json)
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


start()

# Todo: parse all json outputs into a neat PDF for the BT
# Todo: incorporate LangChain's short-term history database support