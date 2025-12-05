
from agents import ListenAgent, ObserveAgent, AdaptAgent, AssistAgent, MentorAgent

def run_session():
    print("--- Starting Dyslexia Reading Assistant Session ---")
    
    # Initialize Agents
    listener = ListenAgent()
    observer = ObserveAgent()
    adapter = AdaptAgent()
    assistant = AssistAgent()
    mentor = MentorAgent()

    # 1. Listen Agent acts
    error_rate, wpm = listener.listen()

    # 2. Observe Agent acts
    focus_score = observer.observe()

    # 3. Adapt Agent acts based on inputs
    next_difficulty = adapter.adapt(error_rate, wpm, focus_score)
    
    # 4. Assist Agent configures the UI for the *next* segment
    assistance_config = assistant.provide_assistance(next_difficulty)
    
    # 5. Mentor Agent provides closing feedback
    feedback = mentor.provide_feedback(error_rate, focus_score)

    print(f"\n--- Session Result ---")
    print(f"Inputs -> Error Rate: {error_rate:.2f}, WPM: {wpm}, Focus: {focus_score:.2f}")
    print(f"Output -> Next Content Difficulty: {next_difficulty:.2f}")
    print(f"Assistance -> {assistance_config}")
    print(f"Mentor -> {feedback}")
    print("----------------------")

if __name__ == "__main__":
    run_session()
