import os
import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# Define your environment variables explicitly
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = "serin-490413"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# Initialize your testing agent
root_agent = Agent(
    name="ExAai_Testing_Agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful testing assistant. Respond politely to tests."
)

# 1. Initialize the Runner to manage the agent
runner = InMemoryRunner(
    agent=root_agent,
    app_name="testing_app",
)

def run_local_test():
    # 2. Create a local session to hold conversation state
    session = asyncio.run(runner.session_service.create_session(
        app_name="testing_app",
        user_id="local_user"
    ))

    # 3. Format the input message using the GenAI types module
    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hello!")]
    )

    print("Sending test message...")
    
    # 4. Run the agent and iterate through the response events
    for event in runner.run(
        user_id="local_user",
        session_id=session.id,
        new_message=new_message,
    ):
        # Extract and print the agent's text response chunks
        if event.content and event.content.parts and event.content.parts[0].text:
            print(event.content.parts[0].text, end="")
    
    print("\n")

if __name__ == "__main__":
    run_local_test()