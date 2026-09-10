"""把探针注入解释器：安装/卸载 ``pprobe.pth``。

``.pth`` 文件里以 ``import`` 开头的行会在解释器启动时被执行，因此只要把这一行放进
任一 ``site-packages`` 目录，``python train.py`` / ``torchrun --nproc_per_node=N`` 拉起的
每个进程都会自动装载探针，无需改动框架源码。

关键设计：那一行用 ``and`` 短路，未设置 ``PPROBE_ENABLE`` 时连 ``pprobe`` 都不会被导入，
对环境中其它 Python 进程零影响。
"""

from __future__ import annotations

import os
import site
import subprocess
import sys
from typing import Any

from .util import log

PTH_NAME = "pprobe.pth"
_PTH_LINE = (
    "import os as _pprobe_os, sys as _pprobe_sys; "
    '_pprobe_os.environ.get("PPROBE_ENABLE", "").strip().lower() in ("1", "true", "yes", "y", "on") '
    'and __import__("pprobe.bootstrap", fromlist=["boot_from_pth"]).boot_from_pth()'
)
_PTH_CONTENT = f"# pprobe: PyTorch 精度探针自动注入（设置 PPROBE_ENABLE=1 才会生效）\n{_PTH_LINE}\n"


def pth_source() -> str:
    return _PTH_LINE


def candidate_dirs(python: str | None = None) -> list[str]:
    """候选 site-packages 目录（当前解释器或指定解释器）。"""
    code = (
        "import json,site,sys,os\n"
        "dirs=[]\n"
        "usp=getattr(site,'getusersitepackages',None)\n"
        "if getattr(site,'ENABLE_USER_SITE',False) and usp:\n"
        "    dirs.append(usp())\n"
        "try:\n"
        "    dirs.extend(site.getsitepackages())\n"
        "except Exception:\n"
        "    pass\n"
        "seen=[]\n"
        "for d in dirs:\n"
        "    if d and d not in seen: seen.append(d)\n"
        "print(json.dumps({'dirs':seen,'prefix':sys.prefix,'executable':sys.executable,"
        "'user_site_enabled':bool(getattr(site,'ENABLE_USER_SITE',False))}))\n"
    )
    if python and os.path.abspath(python) != os.path.abspath(sys.executable):
        try:
            out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
        except Exception as e:
            raise RuntimeError(f"调用 {python} 失败: {e}") from e
        if out.returncode != 0:
            raise RuntimeError(f"调用 {python} 失败: {out.stderr.strip()[:400]}")
        import json

        return json.loads(out.stdout)["dirs"]
    dirs: list[str] = []
    usp = getattr(site, "getusersitepackages", None)
    if getattr(site, "ENABLE_USER_SITE", False) and usp:
        try:
            dirs.append(usp())
        except Exception:
            pass
    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass
    return [d for d in dict.fromkeys(dirs) if d]


def find_existing(python: str | None = None) -> list[str]:
    hits = []
    for d in candidate_dirs(python):
        p = os.path.join(d, PTH_NAME)
        if os.path.isfile(p):
            hits.append(p)
    return hits


def install(python: str | None = None, target_dir: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """写入 ``pprobe.pth``；返回 ``{"path", "dirs", "already"}``。"""
    if target_dir:
        dirs = [os.path.abspath(os.path.expanduser(target_dir))]
    else:
        dirs = candidate_dirs(python)
        # 优先可写目录：用户 site-packages > 环境 site-packages
        dirs = [d for d in dirs if os.path.isdir(d)] or dirs
    written = None
    for d in dirs:
        path = os.path.join(d, PTH_NAME)
        if os.path.isfile(path) and _PTH_LINE in _read(path):
            return {"path": path, "dirs": dirs, "already": True, "writable": True}
        try:
            os.makedirs(d, exist_ok=True)
            if not dry_run:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(_PTH_CONTENT)
            written = path
            break
        except OSError as e:
            log(f"无法写入 {d}: {e}")
            continue
    return {"path": written, "dirs": dirs, "already": False, "writable": written is not None}


def uninstall(python: str | None = None, target_dir: str | None = None) -> list[str]:
    removed = []
    targets = [os.path.join(os.path.abspath(os.path.expanduser(target_dir)), PTH_NAME)] if target_dir else find_existing(python)
    for p in targets:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                if _PTH_LINE not in f.read():
                    log(f"{p} 不是 pprobe 生成的内容，跳过删除")
                    continue
            os.unlink(p)
            removed.append(p)
    return removed


def status(python: str | None = None) -> dict[str, Any]:
    import json

    code = (
        "import json,site,sys,os\n"
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'user_site':bool(getattr(site,'ENABLE_USER_SITE',False)),"
        "'no_user_site':bool(os.environ.get('PYTHONNOUSERSITE')),"
        "'pth_imported':any('pprobe' in getattr(f,'__name__','') or getattr(f,'_pprobe_watcher',False) for f in sys.meta_path)}))\n"
    )
    info: dict[str, Any] = {}
    target = python or sys.executable
    try:
        out = subprocess.run([target, "-c", code], capture_output=True, text=True, timeout=60)
        info = json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr.strip()[:300]}
    except Exception as e:
        info = {"error": repr(e)}
    try:
        dirs = candidate_dirs(python)
    except Exception as e:
        dirs, info["dirs_error"] = [], repr(e)
    existing = find_existing(python)
    pp = os.environ.get("PPROBE_ENABLE")
    return {
        "interpreter": info.get("executable", target),
        "prefix": info.get("prefix", ""),
        "candidate_dirs": dirs,
        "pth_files": existing,
        "installed": bool(existing),
        "user_site_enabled": info.get("user_site"),
        "pythonnousersite": info.get("no_user_site"),
        "enable_env": pp,
        "will_activate": bool(existing) and (pp or "").strip().lower() in ("1", "true", "yes", "y", "on"),
        "self_pth_imported": info.get("pth_imported"),
    }


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""
