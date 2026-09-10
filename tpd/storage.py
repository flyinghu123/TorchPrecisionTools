"""
Storage management for TPD.
Uses JSONL format for efficient append-only writing and easy comparison.
"""

import json
import os
import threading
from typing import Any

from .config import Config


class StorageManager:
    """Manages data persistence in JSONL format."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.rank = Config.RANK
        self._lock = threading.Lock()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # JSONL file path
        self.jsonl_path = os.path.join(self.output_dir, f"rank{self.rank}.jsonl")
        self.stacks_path = os.path.join(self.output_dir, f"stacks_rank{self.rank}.json")
        self.config_path = os.path.join(self.output_dir, f"config_rank{self.rank}.json")

        # Save config
        self._save_config()

    def _save_config(self):
        """Save configuration for this run."""
        with open(self.config_path, "w") as f:
            json.dump(Config.summary(), f, indent=2)

    def append_record(self, record: dict):
        """Append a single record to the JSONL file."""
        with self._lock:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def append_records(self, records: list[dict]):
        """Append multiple records to the JSONL file."""
        with self._lock:
            with open(self.jsonl_path, "a") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def load_records(self, jsonl_path: str | None = None) -> list[dict]:
        """Load all records from a JSONL file."""
        path = jsonl_path or self.jsonl_path
        if not os.path.exists(path):
            return []

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip corrupted lines
        return records

    def get_jsonl_path(self) -> str:
        """Get the path to the JSONL file."""
        return self.jsonl_path

    def get_stacks_path(self) -> str:
        """Get the path to the stacks file."""
        return self.stacks_path


def load_records_from_dir(result_dir: str, rank: int = 0) -> list[dict]:
    """Load records from a result directory for a specific rank."""
    jsonl_path = os.path.join(result_dir, f"rank{rank}.jsonl")
    if not os.path.exists(jsonl_path):
        return []

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_config_from_dir(result_dir: str, rank: int = 0) -> dict:
    """Load configuration from a result directory."""
    config_path = os.path.join(result_dir, f"config_rank{rank}.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f)
