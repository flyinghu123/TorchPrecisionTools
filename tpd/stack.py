"""
Stack trace management with ID-based deduplication.
"""

import hashlib
import json
import os
import traceback
from typing import Optional


class StackManager:
    """Manages stack traces with ID-based deduplication."""

    def __init__(self):
        self._stack_to_id: dict[str, str] = {}
        self._id_to_stack: dict[str, list[str]] = {}
        self._counter = 0

    def get_stack_id(self, frames: Optional[list[str]] = None) -> str:
        """
        Get or create a unique ID for the given stack frames.
        If frames is None, capture current stack.
        Returns a unique ID string.
        """
        if frames is None:
            frames = traceback.format_stack()

        # Create a hash of the stack frames
        stack_str = "\n".join(frames)
        stack_hash = hashlib.md5(stack_str.encode()).hexdigest()[:12]

        if stack_hash in self._stack_to_id:
            return self._stack_to_id[stack_hash]

        # Create new ID
        self._counter += 1
        stack_id = f"S{self._counter:06d}_{stack_hash}"

        self._stack_to_id[stack_hash] = stack_id
        self._id_to_stack[stack_id] = frames

        return stack_id

    def get_stack_by_id(self, stack_id: str) -> Optional[list[str]]:
        """Retrieve full stack trace by ID."""
        return self._id_to_stack.get(stack_id)

    def save_stacks(self, filepath: str):
        """Save all stack traces to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self._id_to_stack, f, indent=2)

    def load_stacks(self, filepath: str):
        """Load stack traces from a JSON file."""
        if not os.path.exists(filepath):
            return
        with open(filepath) as f:
            self._id_to_stack = json.load(f)
        self._stack_to_id = {}
        for stack_id, frames in self._id_to_stack.items():
            stack_str = "\n".join(frames)
            stack_hash = hashlib.md5(stack_str.encode()).hexdigest()[:12]
            self._stack_to_id[stack_hash] = stack_id
        if self._id_to_stack:
            max_num = max(int(sid.split("_")[0][1:]) for sid in self._id_to_stack)
            self._counter = max_num


# Global instance
_global_stack_manager = StackManager()


def get_stack_manager() -> StackManager:
    """Get the global stack manager instance."""
    return _global_stack_manager


def capture_current_stack(skip_frames: int = 2) -> list[str]:
    """Capture current stack trace, skipping internal frames."""
    frames = traceback.format_stack()
    # Skip internal TPD frames
    filtered = [f for f in frames if "/tpd/" not in f]
    return filtered
