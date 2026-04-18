"""
Integration test for Gemini CLI Provider with Tool Use.
"""
import asyncio
import os
import shutil
from pathlib import Path

# Ensure we're using the project root
os.chdir(Path(__file__).parent.parent)

from src.predacore.config import PredaCoreConfig, LLMConfig
from src.predacore.core import PredaCoreCore
from src.predacore.sessions import Session

async def run_test():
    print("Checking for gemini CLI...")
    if not shutil.which("gemini"):
        print("SKIPPING: 'gemini' CLI not found in PATH.")
        return

    print("Initializing PredaCore with gemini-cli provider...")
    
    # Configure PredaCore to use Gemini CLI
    config = PredaCoreConfig(
        llm=LLMConfig(
            provider="gemini-cli",
            model="gemini-2.0-pro-exp-02-05",  # The requested model
            temperature=0.7
        ),
        home_dir="/tmp/predacore_test_home"
    )
    
    # Initialize Core
    core = PredaCoreCore(config)
    session = Session("test_session_id")

    # Define the prompt
    prompt = (
        "Please check the current working directory using 'list_directory'. "
        "Then, create a file named 'hello_gemini.txt' with the content: "
        "'Hello from Gemini 3 Pro via CLI!'. "
        "Finally, read the file back to confirm."
    )
    
    print(f"\n[User]: {prompt}\n")
    
    # Run the agent loop
    # We use a mocked stream_fn to print output in real-time
    async def print_stream(token: str):
        print(token, end="", flush=True)

    response = await core.process(
        user_id="tester",
        message=prompt,
        session=session,
        stream_fn=print_stream
    )
    
    print(f"\n\n[Final Response]: {response}")

    # Verification
    expected_file = Path("hello_gemini.txt")
    if expected_file.exists():
        content = expected_file.read_text()
        print(f"\nSUCCESS: File created with content: '{content}'")
        # Cleanup
        expected_file.unlink()
    else:
        print("\nFAILURE: File was not created.")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
