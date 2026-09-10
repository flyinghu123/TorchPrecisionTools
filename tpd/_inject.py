"""
Injection entry point for .pth file mechanism.
This file is imported by the .pth file when Python starts.
"""

import os
import sys

# Only proceed if TPD is enabled
if os.environ.get("TPD_ENABLED", "0") == "1":
    try:
        # Import the main TPD package to trigger initialization
        import tpd
    except ImportError as e:
        print(f"[TPD] Warning: Failed to import tpd: {e}", file=sys.stderr)
        print(f"[TPD] Make sure tpd is installed: pip install -e /path/to/torch-precision-debugger", file=sys.stderr)
    except Exception as e:
        print(f"[TPD] Error during injection: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
