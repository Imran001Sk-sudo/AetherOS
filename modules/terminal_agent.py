import subprocess
import sys
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from core.safety_gate import analyze_code_safety

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Enforce isolated execution directory
WORKSPACE_DIR = Path.cwd() / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

def run_sandboxed_code(code: str, max_retries: int = 3) -> dict:
    """
    Validates code against the AST safety gate, writes it to a designated workspace,
    and executes it in an isolated subprocess with automatic error patching.
    """
    current_code = code

    for attempt in range(1, max_retries + 1):
        # 1. AST Static Gate Check
        is_safe, violations = analyze_code_safety(current_code)
        if not is_safe:
            return {
                "success": False,
                "output": f"Security Policy Violation: {'; '.join(violations)}",
                "code": current_code
            }

        # 2. Write to isolated workspace temp script
        temp_file = WORKSPACE_DIR / "agent_task.py"
        temp_file.write_text(current_code, encoding="utf-8")

        # 3. Execute isolated subprocess within workspace
        try:
            process = subprocess.run(
                [sys.executable, str(temp_file)],
                cwd=str(WORKSPACE_DIR),
                capture_output=True,
                text=True,
                timeout=30
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Execution timed out (30s limit exceeded).", "code": current_code}

        if process.returncode == 0:
            return {
                "success": True,
                "output": process.stdout.strip() or "[Success: Process exited cleanly with no stdout]",
                "code": current_code
            }

        # If failed and attempts remain, self-heal
        if attempt == max_retries:
            return {
                "success": False,
                "output": f"Execution failed after {max_retries} attempts.\nLast error:\n{process.stderr.strip()}",
                "code": current_code
            }

        prompt = (
            "The following Python code failed. Fix it and return ONLY executable Python code in raw text.\n\n"
            f"Code:\n{current_code}\n\n"
            f"Traceback:\n{process.stderr.strip()}"
        )

        response = client.chat.completions.create(
            model="gemini-3.6-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        current_code = response.choices[0].message.content.strip()
        if current_code.startswith("```"):
            lines = current_code.splitlines()
            current_code = "\n".join([l for l in lines if not l.strip().startswith("```")])

    return {"success": False, "output": "Max retries exceeded.", "code": current_code}