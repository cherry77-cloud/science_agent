import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from agno.tools import tool
from config import GENERATED_DATA_DIR


@tool(name="execute_python_code_tool")
def execute_code(code_str: str, timeout: int = 1800) -> dict:
    """
    Saves the given Python code to a temporary file and executes it
    as a separate process, capturing its stdout and stderr using text mode.

    Args:
        code_str (str): The Python code to execute.
        timeout (int): Timeout in seconds for the execution.

    Returns:
        dict: A dictionary containing:
            - 'exit_code': The exit code of the script (0 for success).
            - 'stdout': The captured standard output (string).
            - 'stderr': The captured standard error (string).
    """
    stdout_str = ""
    stderr_str = ""

    try:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            script_path = tmpdir / "experiment_script.py"

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            process = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=tmpdir,
                check=False,
                env=env,
            )

            stdout_str = (process.stdout if process.stdout is not None else "").strip()
            stderr_str = (process.stderr if process.stderr is not None else "").strip()

    except subprocess.TimeoutExpired:
        stderr_str = f"Execution timed out after {timeout} seconds."
    except FileNotFoundError:
        stderr_str = (
            "Python interpreter not found. Ensure Python is in your system PATH."
        )
    except Exception as e:
        stderr_str = f"An unexpected error occurred in execute_code tool: {str(e)}"

    return {
        "stdout": stdout_str,
        "stderr": stderr_str,
    }


def execute_python_code_simple(
    code_string: str,
    dataset_path_str: Optional[str] = None,
    target_output_dir: Path = GENERATED_DATA_DIR,
) -> Tuple[str, str]:
    stdout_str = ""
    stderr_str = ""

    escaped_dataset_path = (
        dataset_path_str.replace("\\", "\\\\") if dataset_path_str else ""
    )
    escaped_output_dir = str(target_output_dir.resolve()).replace("\\", "\\\\")

    script_header = (
        f"import pandas as pd\n"
        f"import numpy as np\n"
        f"import matplotlib.pyplot as plt\n"
        f"import seaborn as sns\n"
        f"from pathlib import Path\n"
        f"from uuid import uuid4\n\n"
        f"current_dataset_path = r'{escaped_dataset_path}' if '{escaped_dataset_path}' else None\n"
        f"output_dir_path = Path(r'{escaped_output_dir}')\n"
        f"output_dir_path.mkdir(parents=True, exist_ok=True)\n\n"
    )

    full_code_to_execute = script_header + code_string

    try:
        process = subprocess.Popen(
            ["python", "-c", full_code_to_execute],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout_str, stderr_str = process.communicate(timeout=300)

    except subprocess.TimeoutExpired:
        stderr_str += "\nCode execution timed out."
    except Exception as e:
        stderr_str += f"\nError during code execution setup: {str(e)}"

    return stdout_str.strip(), stderr_str.strip()
