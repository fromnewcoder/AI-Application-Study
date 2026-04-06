import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.graph import app

load_dotenv()

def main():
    # 1. Config is REQUIRED for Memory/Interrupts
    config = {"configurable": {"thread_id": "1"}}
    
    # 2. Initial Run
    print("\n--- Starting Graph ---")
    user_input = "Tell me about the thing." # Vague query triggers low score
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    # Run until completion OR interruption
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            print(f"Node '{key}' finished.")

    # 3. Check status
    snapshot = app.get_state(config)
    
    # If the next node is 'rewrite', we hit the breakpoint
    if snapshot.next and "rewrite" in snapshot.next:
        print("\n🛑 GRAPH PAUSED: Low score detected.")
        print(f"Feedback: {snapshot.values['feedback']}")
        
        # 4. Human Decision
        approval = input("\nApprove rewrite? (y/n): ")
        
        if approval.lower() == "y":
            print("\n--- Resuming Graph ---")
            # Passing None resumes execution from the paused state
            for output in app.stream(None, config=config):
                for key, value in output.items():
                    print(f"Node '{key}' finished.")
        else:
            print("--- Process Terminated by User ---")

if __name__ == "__main__":
    main()
