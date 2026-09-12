import os
import sys
import json
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from openai import OpenAI

from modules.terminal_agent import run_with_self_healing
from modules.web_agent import search_and_extract
from core.safety_gate import verify_command_safety

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Executes local Python code with self-healing recovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Raw Python code to run"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Launches a browser session to research web queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def aether_dispatch(name: str, args: dict) -> str:
    if name == "run_python_code":
        code = args.get("code", "")
        if not verify_command_safety(code):
            return "Command execution cancelled: rejected by safety policy."
        return run_with_self_healing(code)

    elif name == "search_web":
        query = args.get("query", "")
        return asyncio.run(search_and_extract(query))

    return f"Unknown tool: {name}"

def safe_chat_completion(messages, retries=4, delay=12):
    """Executes chat completions with automatic handling for 429 RPM limits."""
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1
            )
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[AetherOS] Rate limit hit (Free tier 5 RPM). Pausing {delay}s before retry (Attempt {attempt}/{retries})...")
                time.sleep(delay)
                if attempt == retries:
                    raise e
            else:
                raise e

def run_aether_loop(goal: str):
    print(f"\n==========================================")
    print(f"[AetherOS Planner] New Goal: {goal}")
    print(f"==========================================")

    messages = [
        {
            "role": "system",
            "content": (
                "You are AetherOS, an autonomous multi-agent system operator. "
                "You break down user objectives, dispatch web research tools, and "
                "write and execute self-healing local Python scripts to complete tasks."
            )
        },
        {"role": "user", "content": goal}
    ]

    for step in range(1, 6):
        print(f"\n[AetherOS Step {step}] Reasoning next action...")
        response = safe_chat_completion(messages)

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"\n[AetherOS Task Complete]\n{msg.content}")
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"[AetherOS Action] Calling tool '{fn_name}'")
            tool_result = aether_dispatch(fn_name, fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

if __name__ == "__main__":
    print("\n=== Welcome to AetherOS Autonomous Operator ===")
    print("Type your high-level goal, or press Enter for the default end-to-end demo.")
    user_goal = input("\nGoal > ").strip()
    
    if not user_goal:
        user_goal = (
            "Search the web for the latest updates on Python 3.14 features, "
            "then write and execute a local Python script that creates a file named "
            "'python_summary.txt' containing the findings."
        )

    run_aether_loop(user_goal)