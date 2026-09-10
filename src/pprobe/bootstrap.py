"""解释器启动阶段的引导：``.pth`` 入口 + torch 懒加载监听。

``pprobe.pth`` 会在每个 Python 进程启动时执行一行 ``import``，因此这里必须做到：
* 未设置 ``PPROBE_ENABLE`` 时**不导入 torch、不建目录、不装信号**，接近零开销；
* 不在启动期导入 torch（很多进程只是 import torch 前就要 import 别的包），
  改为在 ``sys.meta_path`` 上挂一个监听器，等 ``torch`` 真正被导入完再装 hook，
  这样 ``torchrun --nproc_per_node=N`` 拉起的每个 rank 进程都会自动生效。
"""

from __future__ import annotations

import atexit
import importlib
import importlib.machinery
import importlib.util
import os
import sys
from typing import Any, Callable

from .util import log

_WATCH_TARGETS = ("torch", "torch.nn")
_booted = False
_watcher: "_ImportWatcher | None" = None


def is_enabled_by_env(env: dict[str, str] | None = None) -> bool:
    import os as _os

    e = _os.environ if env is None else env
    return (e.get("PPROBE_ENABLE") or "").strip().lower() in ("1", "true", "yes", "y", "on")


def boot_from_pth() -> None:
    """.pth 文件调用的入口：任何异常都不能影响解释器启动。"""
    global _booted
    if _booted or not is_enabled_by_env():
        return
    try:
        _booted = True
        start()
    except BaseException as e:  # noqa: BLE001
        log(f"初始化失败（已忽略，训练继续）: {type(e).__name__}: {e}")
        if os.environ.get("PPROBE_TRACEBACK"):
            import traceback

            traceback.print_exc()


def init(**overrides: Any):
    """手动注入入口：``import pprobe; pprobe.init()``（不需要 .pth 时使用）。"""
    global _booted
    _booted = True
    return start(overrides=overrides or None, force=True)


def start(overrides: dict | None = None, force: bool = False):
    from .config import Config

    cfg = Config.from_env()
    if overrides:
        cfg = cfg.replace(**overrides)
    if not force and not cfg.enable:
        return None
    if not cfg.enable:  # 显式 init() 时即使没设环境变量也要工作
        cfg.enable = True

    from .recorder import get_recorder, init_recorder

    rec = get_recorder()
    if rec is not None:
        return rec
    if cfg.verbose:
        log(f"启用（PPROBE_ENABLE）→ 输出目录 {os.path.abspath(os.path.expanduser(cfg.out_dir))}，{cfg.banner()}")
    _warn_unknown_env()
    _setup_fault_handler(cfg)
    rec = init_recorder(cfg)
    _enable_debug_env(cfg, rec)
    _install_when_torch_ready(rec, cfg)
    return rec


# ----------------------------------------------------------------------
def _install_when_torch_ready(rec, cfg) -> None:
    if "torch" in sys.modules and getattr(sys.modules["torch"], "nn", None) is not None:
        _install_hooks(rec, cfg)
        return
    global _watcher
    if _watcher is None:
        _watcher = _ImportWatcher("torch", lambda: _on_torch_imported(rec, cfg))
        _watcher.register()
    atexit.register(_warn_if_never_used, rec)


def _on_torch_imported(rec, cfg) -> None:
    try:
        import torch  # noqa: F401

        if getattr(torch, "nn", None) is None:  # 老版本/惰性导入：再等 torch.nn
            w = _ImportWatcher("torch.nn", lambda: _install_hooks(rec, cfg))
            w.register()
            return
        _install_hooks(rec, cfg)
    except BaseException as e:
        log(f"torch 导入后装 hook 失败: {type(e).__name__}: {e}")


def _install_hooks(rec, cfg) -> None:
    try:
        from . import hooks

        hooks.install(rec)
        rec._hooks_installed = True
    except BaseException as e:
        rec.note_error(f"install:{type(e).__name__}")
        log(f"安装 hook 失败: {type(e).__name__}: {e}")


def _warn_if_never_used(rec) -> None:
    try:
        if not rec._hooks_installed and rec.cfg.verbose and not rec._child:
            log("进程结束前 torch 仍未被导入（或 torch.nn 不可用），本次没有采集任何数据")
    except Exception:
        pass


def _warn_unknown_env() -> None:
    """拼错的环境变量名最容易被忽略（设了没生效），启动时直接提示。"""
    try:
        from .config import unknown_env

        bad = unknown_env()
        if bad:
            log(f"忽略无法识别的环境变量 {bad}（用 `pprobe env` 查看合法名单）")
    except Exception:
        pass


def _setup_fault_handler(cfg) -> None:
    if not cfg.faulthandler:
        return
    try:
        import faulthandler

        faulthandler.enable(all_threads=True)
    except Exception as e:
        log(f"faulthandler 启用失败: {e!r}")


def _enable_debug_env(cfg, rec) -> None:
    """把随机采样实际使用的种子回写进环境，方便子进程/复现脚本对齐。"""
    if cfg.sample_mode in ("random",) and cfg.sample_seed is None:
        os.environ["PPROBE_SEED_USED"] = str(rec.sample_seed_used)


# ----------------------------------------------------------------------
class _ImportWatcher:
    """在指定模块导入完成后执行回调（不改变正常导入语义，只包一层 loader）。"""

    _pprobe_watcher = True

    def __init__(self, name: str, callback: Callable[[], None]):
        self.name = name
        self.callback = callback
        self._busy = False
        self._done = False

    def register(self) -> None:
        for f in sys.meta_path:
            if getattr(f, "_pprobe_watcher", False) and getattr(f, "name", None) == self.name:
                return
        sys.meta_path.insert(0, self)

    def detach(self) -> None:
        self._done = True
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

    # -- MetaPathFinder 协议 --
    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname != self.name or self._busy or self._done:
            return None
        spec = self._real_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        inner = spec.loader
        loader = _LoaderProxy(inner, self._after_exec)
        spec.loader = loader
        return spec

    def _real_spec(self, fullname: str, path) -> Any:
        self._busy = True
        try:
            if "." in fullname:
                parent = fullname.rpartition(".")[0]
                if parent not in sys.modules:
                    return None  # 父包尚未导入，交给下一次触发
                return importlib.util.find_spec(fullname)
            return importlib.machinery.PathFinder.find_spec(fullname, sys.path)
        finally:
            self._busy = False

    def _after_exec(self, module) -> None:
        self.detach()
        try:
            self.callback()
        except BaseException as e:
            log(f"模块 {self.name} 导入回调异常: {type(e).__name__}: {e}")


class _LoaderProxy:
    """透明转发真实 loader，只在 exec_module 完成后追加一个回调。"""

    def __init__(self, inner, after: Callable[[Any], None]):
        self._inner = inner
        self._after = after

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            self._after(module)
        finally:
            pass

    def __getattr__(self, item):
        return getattr(self._inner, item)
