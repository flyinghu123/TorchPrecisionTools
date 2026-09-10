"""
Setup script for TPD - Torch Precision Debugger.
Handles .pth file installation for auto-injection.
"""

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Custom install command to create .pth file in site-packages."""

    def run(self):
        # Run the standard install first
        install.run(self)

        # Create .pth file in site-packages
        site_packages = self.install_lib
        pth_file = os.path.join(site_packages, "tpd.pth")

        # .pth file content: execute code to import tpd._inject if TPD_ENABLED=1
        pth_content = 'import os; exec("if os.environ.get(\\"TPD_ENABLED\\", \\"0\\") == \\"1\\":\\n try:\\n  import tpd._inject\\n except Exception: pass")'

        with open(pth_file, "w") as f:
            f.write(pth_content)

        print(f"[TPD] Created {pth_file} for auto-injection")
        print(f"[TPD] Set TPD_ENABLED=1 to activate")


setup(
    name="torch-precision-debugger",
    version="0.1.0",
    packages=find_packages(),
    cmdclass={
        "install": CustomInstallCommand,
    },
    entry_points={
        "console_scripts": [
            "tpd=tpd.cli:main",
        ],
    },
    install_requires=[
        "torch>=1.12.0",
        "numpy",
    ],
    python_requires=">=3.10",
)
