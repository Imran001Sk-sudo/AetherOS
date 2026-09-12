import sys
from pathlib import Path
import os
import json
import time
import asyncio
import streamlit as st
from openai import OpenAI

# Ensure root directory is in sys.path so modules resolve correctly on Linux / Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Optional import: load .env locally if available; ignored on cloud deployments
try:
    from dotenv import load_dotenv
    load_dotenv()
except (ImportError, ModuleNotFoundError):
    pass

from modules.terminal_agent import run_sandboxed_code, WORKSPACE_DIR
from modules.web_agent import search_and_extract

st.set_page_config(page_title="AetherOS | Autonomous Agent", page_icon="⚡", layout="wide")

st.title("⚡ AetherOS Production Console")
st.caption("Deterministic Multi-Agent Engine • Playwright Scraping • AST Sandboxed Execution")

# Check Streamlit Cloud Secrets first, fallback to local environment variables
api_key = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Active Gemini models with operational quotas
MODEL_TIERS = ["gemini-3.6-flash", "gemini-2.5-flash"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Executes local Python code inside an isolated workspace sandbox with self-healing.",
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
            "description": "Searches the live web and synthesizes findings using Playwright.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def request_completion_with_fallback(messages, tools=None):
    """Executes calls using active models with automatic fallback on rate limits."""
    last_error = None
    for model_name in MODEL_TIERS:
        for attempt in range(2):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                
                return client.chat.completions.create(**kwargs), model_name
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "404" in err_str:
                    break
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(6)
                    continue
                raise e
    raise last_error

# Sidebar Controls & File Browser
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("Core Engine: Ready")
    st.info(f"Sandbox Directory:\n`{WORKSPACE_DIR}`")
    
    if st.button("🔄 Clear / New Goal", use_container_width=True):
        st.rerun()

    st.header("📁 Workspace Files")
    files = list(WORKSPACE_DIR.glob("*"))
    if files:
        for f in files:
            st.code(f.name)
    else:
        st.write("Workspace currently clean.")

# User Goal Input Area
user_goal = st.text_area(
    "Set Autonomous Goal",
    placeholder="e.g., Research Vishnu Institute of Technology, extract its departments, and save a summary report to the workspace."
)

start_btn = st.button("🚀 Execute Goal", type="primary", use_container_width=True)

if start_btn and user_goal:
    messages = [
        {"role": "system", "content": "You are AetherOS, a production autonomous agent. Complete user objectives by searching the web and running local sandboxed scripts."},
        {"role": "user", "content": user_goal}
    ]

    log_box = st.container()

    for step in range(1, 6):
        with log_box:
            st.subheader(f"Step {step}: Reasoning")

        try:
            response, used_model = request_completion_with_fallback(messages, tools=TOOLS)
        except Exception as e:
            st.error(f"Quota Exceeded across all fallback models: {e}")
            break

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            st.balloons()
            st.success(f"### Task Completed (via {used_model})")
            st.markdown(msg.content)
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "search_web":
                with log_box:
                    st.info(f"🌐 **Web Agent**: Searching for `{fn_args.get('query')}`")
                    result = asyncio.run(search_and_extract(fn_args.get("query")))
                    st.text_area("Extracted Search Data", result, height=120)

            elif fn_name == "run_python_code":
                code = fn_args.get("code", "")
                with log_box:
                    st.warning("💻 **Terminal Agent**: Running Sandboxed Python Script")
                    st.code(code, language="python")
                    
                    exec_result = run_sandboxed_code(code)
                    if exec_result["success"]:
                        st.success(f"Output:\n{exec_result['output']}")
                    else:
                        st.error(f"Failed:\n{exec_result['output']}")
                    result = exec_result["output"]

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        time.sleep(3)