import ast
import os
import resource
import subprocess
import sys
import tempfile
import time

# ==============================================================================
# SOTA SLM Sandbox & Execution Guard
# ==============================================================================

class SecureASTVisitor(ast.NodeVisitor):
    """
    Besucher für den Abstract Syntax Tree des generierten Codes.
    Schlägt Alarm (ValueError), wenn schädliche Aufrufe oder Imports detektiert werden.
    """
    
    FORBIDDEN_BUILTINS = {
        'eval', 'exec', 'compile', 'open', '__import__',
        'getattr', 'setattr', 'delattr', 'globals', 'locals',
        'input', 'breakpoint'
    }

    FORBIDDEN_MODULE_CALLS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
        "signal",
    }
    
    # Positive allowlist to keep the sandbox predictable and block fs/network modules.
    ALLOWED_IMPORTS = {
        'math', 'itertools', 'functools', 'collections', 'heapq', 'bisect',
        're', 'string', 'typing', 'dataclasses', 'statistics', 'fractions',
        'decimal', 'random', 'json'
    }

    def visit_Import(self, node):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module not in self.ALLOWED_IMPORTS:
                raise ValueError(f"Forbidden import detected: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module not in self.ALLOWED_IMPORTS:
                raise ValueError(f"Forbidden from-import detected: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Überprüfe direkte Funktionsaufrufe auf verbotene Built-Ins
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_BUILTINS:
                raise ValueError(f"Forbidden builtin call detected: {node.func.id}")
        
        # Überprüfe direkte Aufrufe bekannter Gefahrmodule (z.B. os.system).
        # Wir blockieren NICHT allgemein Attribute auf beliebigen Variablen,
        # da sonst legitime Methodenaufrufe (list.append, str.lower, etc.) brechen.
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in self.FORBIDDEN_MODULE_CALLS:
                    raise ValueError(f"Forbidden module call detected: {node.func.value.id}.{node.func.attr}")
        
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Block dunder-attribute traversal used in many Python sandbox escapes.
        if node.attr.startswith("__"):
            raise ValueError(f"Forbidden attribute access detected: {node.attr}")
        self.generic_visit(node)


def is_safe_code(code_string):
    """
    Prüft, ob der Code-String sicher auszuführen ist (AST-Ebene).
    Gibt (True, None) bei Erfolg oder (False, Error_Message) zurück.
    """
    try:
        tree = ast.parse(code_string)
        visitor = SecureASTVisitor()
        visitor.visit(tree)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except ValueError as e:
        return False, f"Security Violation: {e}"
    except Exception as e:
        return False, f"AST Parse Error: {e}"


def run_code_in_sandbox_detailed(code_string, timeout=2.0):
    """
    Führt den Code stark limitiert (hinsichtlich Zeit und Built-ins) aus.
    Gibt ein Dict mit Reward, Timing und Fehlerdetails zurück.
    """
    
    # 1. AST Sicherheitsprüfung (Verhindert Reward-Hacking & Systemmanipulation)
    is_safe, error_msg = is_safe_code(code_string)
    if not is_safe:
        if "Syntax Error" in error_msg:
             return {
                 "score": 0.0,
                 "exec_time": 0.0,
                 "stdout": "",
                 "stderr": error_msg,
                 "error_type": "syntax",
             }
        else:
             return {
                 "score": -1.0,
                 "exec_time": 0.0,
                 "stdout": "",
                 "stderr": error_msg,
                 "error_type": "security",
             }

    # 2. Physikalische Ausführung im isolierten Subprozess
    def _limit_process_resources():
        cpu_seconds = max(1, int(timeout))
        memory_limit = 512 * 1024 * 1024
        file_limit = 1 * 1024 * 1024
        fd_limit = 32
        limits = [
            (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1)),
            (resource.RLIMIT_AS, (memory_limit, memory_limit)),
            (resource.RLIMIT_FSIZE, (file_limit, file_limit)),
            (resource.RLIMIT_NOFILE, (fd_limit, fd_limit)),
        ]
        for limit_name, values in limits:
            try:
                resource.setrlimit(limit_name, values)
            except (ValueError, OSError):
                # Some platforms/sandboxes do not allow all limits.
                continue

    sandbox_env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    }

    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix="slm_sandbox_") as temp_dir:
        def _run_subprocess(use_preexec):
            kwargs = {
                "args": [sys.executable, "-I", "-S", "-c", code_string],
                "capture_output": True,
                "text": True,
                "timeout": timeout,
                "cwd": temp_dir,
                "env": sandbox_env,
            }
            if use_preexec:
                kwargs["preexec_fn"] = _limit_process_resources
            return subprocess.run(**kwargs)

        try:
            try:
                result = _run_subprocess(use_preexec=True)
            except Exception as e:
                if "preexec_fn" not in str(e):
                    raise
                # Fall back to execution without preexec limits if the runtime blocks preexec_fn.
                result = _run_subprocess(use_preexec=False)
            exec_time = time.time() - start_time
        
            # 3. Dense Reward Zuweisung basierend auf Ausbeute
            if result.returncode == 0:
                return {
                    "score": 2.0,
                    "exec_time": exec_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error_type": "",
                }
            elif "AssertionError" in result.stderr:
                return {
                    "score": 0.5,
                    "exec_time": exec_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error_type": "assertion",
                }
            else:
                return {
                    "score": 0.1,
                    "exec_time": exec_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error_type": "runtime",
                }
            
        except subprocess.TimeoutExpired:
            return {
                "score": -0.5,
                "exec_time": timeout,
                "stdout": "",
                "stderr": "Execution timed out.",
                "error_type": "timeout",
            }
        except Exception as e:
            print(f"Sandbox fatal error: {e}")
            return {
                "score": 0.0,
                "exec_time": 0.0,
                "stdout": "",
                "stderr": str(e),
                "error_type": "sandbox_fatal",
            }


def run_code_in_sandbox(code_string, timeout=2.0):
    """
    Backward-compatible wrapper returning (score, exec_time).
    """
    details = run_code_in_sandbox_detailed(code_string, timeout=timeout)
    return details["score"], details["exec_time"]
