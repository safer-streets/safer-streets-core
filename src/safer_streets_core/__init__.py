import importlib.metadata
import os
from pathlib import Path

from dotenv import load_dotenv

__version__ = importlib.metadata.version("safer-streets-core")

load_dotenv()
DATA_DIR = Path(os.getenv("SAFER_STREETS_DATA_DIR", "./data"))
