"""
Signal handling for graceful shutdown.
Ensures data is saved when the program is interrupted.
"""

import atexit
import signal
import sys
from typing import Callable


class SignalHandler:
    """Handles signals to ensure data is saved before exit."""

    def __init__(self):
        self._cleanup_callbacks: list[Callable] = []
        self._original_handlers = {}
        self._installed = False

    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback to be called on exit."""
        self._cleanup_callbacks.append(callback)

    def install(self):
        """Install signal handlers and atexit hook."""
        if self._installed:
            return

        # Save original signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                # Some signals can't be caught in certain contexts
                pass

        # Register atexit hook
        atexit.register(self._atexit_handler)

        self._installed = True

    def uninstall(self):
        """Restore original signal handlers."""
        if not self._installed:
            return

        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass

        self._installed = False

    def _signal_handler(self, signum, frame):
        """Handle incoming signals."""
        print(f"\n[TPD] Received signal {signum}, saving data...", file=sys.stderr)
        self._run_cleanup()

        # Restore original handler and re-raise signal
        original = self._original_handlers.get(signum)
        if original and callable(original):
            # Restore and re-raise
            signal.signal(signum, original)
            import os
            os.kill(os.getpid(), signum)
        else:
            sys.exit(128 + signum)

    def _atexit_handler(self):
        """Handle normal program exit."""
        self._run_cleanup()

    def _run_cleanup(self):
        """Run all cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[TPD] Warning: Cleanup callback failed: {e}", file=sys.stderr)


# Global instance
_global_signal_handler = SignalHandler()


def get_signal_handler() -> SignalHandler:
    """Get the global signal handler instance."""
    return _global_signal_handler
