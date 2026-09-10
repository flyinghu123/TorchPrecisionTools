"""
TPD - Torch Precision Debugger
Configuration via environment variables.
"""

import os


class Config:
    """All configuration is read from environment variables."""

    # === Core switches ===
    ENABLED: bool = os.environ.get("TPD_ENABLED", "0") == "1"

    # === Output ===
    OUTPUT_DIR: str = os.environ.get("TPD_OUTPUT_DIR", "./tpd_results")

    # === Sampling ===
    SAMPLE_COUNT: int = int(os.environ.get("TPD_SAMPLE_COUNT", "50"))
    SAMPLE_MODE: str = os.environ.get("TPD_SAMPLE_MODE", "uniform")  # "uniform" or "random"
    SAMPLE_SEED: int | None = (
        int(os.environ["TPD_SAMPLE_SEED"]) if os.environ.get("TPD_SAMPLE_SEED", "") != "" else None
    )

    # === Limits ===
    MAX_STEPS: int = int(os.environ.get("TPD_MAX_STEPS", "0"))  # 0 = unlimited
    SAVE_INTERVAL: int = int(os.environ.get("TPD_SAVE_INTERVAL", "100"))  # save every N hook calls

    # === Module filter (optional, comma-separated module name prefixes) ===
    MODULE_FILTER: str = os.environ.get("TPD_MODULE_FILTER", "")  # empty = all modules

    # === Distributed ===
    RANK: int = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    WORLD_SIZE: int = int(os.environ.get("WORLD_SIZE", "1"))

    @classmethod
    def get_module_filter_list(cls) -> list[str]:
        if not cls.MODULE_FILTER:
            return []
        return [m.strip() for m in cls.MODULE_FILTER.split(",") if m.strip()]

    @classmethod
    def summary(cls) -> dict:
        return {
            "enabled": cls.ENABLED,
            "output_dir": cls.OUTPUT_DIR,
            "sample_count": cls.SAMPLE_COUNT,
            "sample_mode": cls.SAMPLE_MODE,
            "sample_seed": cls.SAMPLE_SEED,
            "max_steps": cls.MAX_STEPS,
            "save_interval": cls.SAVE_INTERVAL,
            "module_filter": cls.MODULE_FILTER,
            "rank": cls.RANK,
            "world_size": cls.WORLD_SIZE,
        }
