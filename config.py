import os
from dotenv import load_dotenv
from pathlib import Path



OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://xiaoai.plus/v1"
OPENAI_MODEL_NAME = "gpt-4o"
OUTPUT_DIR = Path(r"./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PLOT_OUTPUT_DIR = Path(f"{OUTPUT_DIR}/plots")
DATA_PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATED_DATA_DIR = Path(f"{OUTPUT_DIR}/generated_datasets")
GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
