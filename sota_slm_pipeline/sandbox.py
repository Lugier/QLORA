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
        
        # Überprüfe auf Methodenaufrufe (z.B. os.system, os.popen)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id not in self.ALLOWED_IMPORTS:
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


def run_code_in_sandbox(code_string, timeout=2.0):
    """
    Führt den Code stark limitiert (hinsichtlich Zeit und Built-ins) aus.
    Gibt ein Tuple (Reward Score, Execution Time) zurück.
    Das ermöglicht "Dense Rewards" abhängig von Laufzeiteffizienz.
    """
    
    # 1. AST Sicherheitsprüfung (Verhindert Reward-Hacking & Systemmanipulation)
    is_safe, error_msg = is_safe_code(code_string)
    if not is_safe:
        if "Syntax Error" in error_msg:
             return 0.0, 0.0 # Dense Reward: 0.0, Code ist unleserlich
        else:
             return -1.0, 0.0 # Cheat / Hack-Versuch

    # 2. Physikalische Ausführung im isolierten Subprozess
    def _limit_process_resources():
        cpu_seconds = max(1, int(timeout))
        memory_limit = 512 * 1024 * 1024
        file_limit = 1 * 1024 * 1024
        fd_limit = 32

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        resource.setrlimit(resource.RLIMIT_NOFILE, (fd_limit, fd_limit))

    sandbox_env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    }

    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix="slm_sandbox_") as temp_dir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", code_string],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
                env=sandbox_env,
                preexec_fn=_limit_process_resources,
            )
            exec_time = time.time() - start_time
        
            # 3. Dense Reward Zuweisung basierend auf Ausbeute
            if result.returncode == 0:
                return 2.0, exec_time # Perfekte Ausführung
            elif "AssertionError" in result.stderr:
                return 0.5, exec_time # Dense Reward: AST war okay, lief los, aber Logik-Fehler in Edge-Cases
            else:
                return 0.1, exec_time # Dense Reward: Konnte starten, brach aber wegen Runtime-Error ab
            
        except subprocess.TimeoutExpired:
            return -0.5, timeout # Dense Penalty: O(n^2) endlose Laufzeit
        except Exception as e:
            print(f"Sandbox fatal error: {e}")
            return 0.0, 0.0
