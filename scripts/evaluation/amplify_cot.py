#!/usr/bin/env python3
from pathlib import Path
import sys

# Ensure repo root is on sys.path for package-style import
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.cot_amp.amplify_cot import main  # type: ignore

if __name__ == "__main__":
    main()
