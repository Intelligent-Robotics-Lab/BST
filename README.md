# BST Program

This Behavior Skills Training (BST) program uses the Furhat and NAO social robots to simulate and assess Applied Behavior Analysis (ABA) Discrete Trial Training (DTT) sessions for children with ASD. It recognizes verbal and non-verbal inputs from an interventionist, tracking their state in the training and giving feedback on protocol adherence.


## Phases of BST

- **Introduction**: Furhat briefly explains the DTT protocol to the interventionist.
- **Demonstration**: Furhat demonstrates ideal DTT performance with NAO.
- **Rehearsal**: The interventionist performs DTT on NAO while Furhat evaluates.
- **Feedback**: Furhat relays personalized feedback to the interventionist and generates a viewable PDF of the interventionist's performance.


## Files

- **Main.py**: Core program logic responsible for creating and running sessions and trials.
- **LookupTables.py**: Dictionaries that store ideal phrasings of instructions, reinforcements, prompts, and error corrections. The LLM compares the interventionists' semantics to the ideal phrasings in this file to determine if they match or are synonymous.
- **Vision.py**: Algorithms that check for limb locations and facial expressions, as needed in certain trials.

  
## Prerequisites

- **Furhat social robot**  (with support for Remote API)
- **NAO social robot** 
- **Python 3.9+** 
  - **Flask** (to connect to NAO)
  - **OpenCV 2** (for tracking limb movements)
  - **MediaPipe**
  - **DeepFace**
- **OpenAI API key** (to power the LLM evaluations)
