"""
TPD - Torch Precision Debugger
Auto-injection entry point.

When TPD_ENABLED=1, this module is automatically loaded and hooks all PyTorch modules.
"""

import os
import sys

# Check if TPD is enabled
if os.environ.get("TPD_ENABLED", "0") != "1":
    # Not enabled, do nothing
    pass
else:
    try:
        import torch
        import torch.nn as nn

        from .config import Config
        from .hooks import HookEngine, ModuleCreationMonitor
        from .signals import get_signal_handler
        from .stack import get_stack_manager
        from .storage import StorageManager

        # Initialize components
        storage = StorageManager()
        engine = HookEngine(storage)
        signal_handler = get_signal_handler()

        # Register cleanup
        def _cleanup():
            try:
                engine.finalize()
                engine.remove_hooks()
                stack_manager = get_stack_manager()
                stack_manager.save_stacks(storage.get_stacks_path())
            except Exception as e:
                print(f"[TPD] Error during cleanup: {e}", file=sys.stderr)

        signal_handler.register_cleanup(_cleanup)
        signal_handler.install()

        # Hook all existing modules
        # We need to wait for modules to be created, so we use a trace function
        module_monitor = ModuleCreationMonitor(engine)
        module_monitor.start()

        print(
            f"[TPD] Torch Precision Debugger initialized",
            file=sys.stderr,
        )
        print(
            f"[TPD] Output directory: {Config.OUTPUT_DIR}",
            file=sys.stderr,
        )
        print(
            f"[TPD] Sampling: {Config.SAMPLE_COUNT} elements, mode={Config.SAMPLE_MODE}",
            file=sys.stderr,
        )
        print(
            f"[TPD] Max steps: {Config.MAX_STEPS if Config.MAX_STEPS > 0 else 'unlimited'}",
            file=sys.stderr,
        )
        print(
            f"[TPD] Save interval: {Config.SAVE_INTERVAL}",
            file=sys.stderr,
        )
        print(
            f"[TPD] Rank: {Config.RANK}/{Config.WORLD_SIZE}",
            file=sys.stderr,
        )

    except ImportError as e:
        print(f"[TPD] Warning: Failed to import torch: {e}", file=sys.stderr)
        print(f"[TPD] TPD will not be active", file=sys.stderr)
    except Exception as e:
        print(f"[TPD] Error during initialization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
