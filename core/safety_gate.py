import ast
import re

BLOCKED_MODULES = {
    "shutil", "socket", "ctypes", "winreg", "webbrowser"
}

BLOCKED_CALLS = {
    "os.system", "os.popen", "os.spawn", "os.kill", "os.remove", "os.rmdir",
    "subprocess.Popen", "subprocess.call", "builtins.eval", "builtins.exec"
}

class SafetyASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in BLOCKED_MODULES:
                self.violations.append(f"Direct import of restricted module: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in BLOCKED_MODULES:
                self.violations.append(f"Import from restricted module: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node):
        call_name = ""
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            val = node.func.value
            parts = [node.func.attr]
            while isinstance(val, ast.Attribute):
                parts.append(val.attr)
                val = val.value
            if isinstance(val, ast.Name):
                parts.append(val.id)
            call_name = ".".join(reversed(parts))

        for blocked in BLOCKED_CALLS:
            if call_name == blocked or call_name.endswith("." + blocked.split(".")[-1]):
                if call_name in ["os.system", "os.popen", "subprocess.Popen", "subprocess.call"]:
                    self.violations.append(f"Execution of arbitrary shell command: '{call_name}'")

        self.generic_visit(node)

def analyze_code_safety(code: str) -> tuple[bool, list[str]]:
    """Parses code into an AST and detects malicious operations deterministically."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax Parsing Error: {e.msg} at line {e.lineno}"]

    visitor = SafetyASTVisitor()
    visitor.visit(tree)

    if visitor.violations:
        return False, visitor.violations
    return True, []