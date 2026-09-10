"""forward / backward hook 引擎。

注入方式选择「替换 ``torch.nn.Module.__call__``」而非逐实例 ``register_forward_hook``：
* 覆盖所有 Module 实例（包括 hook 安装之前已经创建好的模型，megatron/llamafactory 里
  往往是先 build_model 再进入训练循环，逐实例注册无法覆盖）；
* 一次包装同时拿到 args/kwargs/output 与模块层级路径，避免 pre/post hook 两次穿越；
* 对 ``torch.func.functional_call``、``nn.Sequential``、子类自定义 ``__call__`` 均已验证可见。

backward 有两条互补路径：
* ``register_full_backward_hook``  -> 模块边界上的 ``grad_input[i]`` / ``grad_output[i]``（默认）
* ``Tensor.register_hook``         -> 单个输入/输出张量的梯度（``grad_of:xxx``，用于
  模块无参数、full hook 无法注册或多输入歧义的兜底）
"""

from __future__ import annotations

import os
import threading
import weakref
from typing import Any

import torch

from .events import SlotCollector, collect_args, collect_params, has_nan, structure_sig
from .recorder import PProbeLimitReached, Recorder
from .util import log

_TLS = threading.local()
_STATE: dict[str, Any] = {"installed": False, "rec": None, "cfg": None, "orig_call": None}
_ORIG: dict[str, Any] = {}
_FUNC_CALLS: dict[str, int] = {}
_GRAD_OCCURRENCE: dict[int, int] = {}


class _ModState:
    """挂在模块实例上的探针状态（放在 ``__dict__`` 里，不进入 state_dict）。"""

    __slots__ = ("path", "cls", "fwd_seq", "fwd_stack", "call_index", "rec_call_index",
                 "bwd_registered", "bwd_failed", "params_done")

    def __init__(self, path: str, cls: str):
        self.path = path
        self.cls = cls
        self.fwd_seq: int | None = None
        self.fwd_stack: str | None = None
        self.call_index = 0
        self.rec_call_index: int | None = None   # 本次被记录的那次调用的序号，反向事件要用它
        self.bwd_registered = False
        self.bwd_failed = False
        self.params_done = False

    def bwd_call_index(self) -> int:
        """反向要对齐到“哪一次前向”：``call_index`` 在前向结束时已经进位到下一次了。"""
        return self.call_index if self.rec_call_index is None else self.rec_call_index


def _mod_state(module: "torch.nn.Module", path: str, cls: str) -> _ModState:
    st = module.__dict__.get("_pprobe_state")
    if st is None:
        st = _ModState(path, cls)
        module.__dict__["_pprobe_state"] = st
    else:
        st.path = path
        st.cls = cls
    return st


def _skip_module(module: "torch.nn.Module") -> bool:
    return module.__dict__.get("_pprobe_off", False)


# ----------------------------------------------------------------------
# 模块层级路径
# ----------------------------------------------------------------------
def _child_name(parent: "torch.nn.Module", child: "torch.nn.Module") -> str:
    cache = parent.__dict__.get("_pprobe_child_names")
    if cache is None:
        cache = weakref.WeakKeyDictionary()
        parent.__dict__["_pprobe_child_names"] = cache
    name = cache.get(child)
    if name is None:
        name = _direct_child(parent, child)
        if name is None:
            # ModuleList / Sequential / 自定义容器里的子模块不在 parent._modules 里，
            # 例如 megatron/llamafactory 常见的 self.blocks = nn.ModuleList([...])
            name = _find_nested_name(parent, child) or ""
        if not name:
            name = type(child).__name__.lower() + "#dyn"
        cache[child] = name
    return name


def _direct_child(parent: "torch.nn.Module", child: "torch.nn.Module") -> str | None:
    try:
        for k, v in parent._modules.items():
            if v is child:
                return k
    except Exception:
        pass
    return None


def _find_nested_name(parent: "torch.nn.Module", child: "torch.nn.Module", max_depth: int = 4,
                      budget: int = 3000) -> str | None:
    """广度优先搜子模块树，返回形如 ``blocks.0`` 的相对路径（只算一次，结果会被缓存）。"""
    frontier: list[tuple[str, "torch.nn.Module", int]] = [("", parent, 0)]
    seen = 0
    while frontier and seen < budget:
        prefix, node, depth = frontier.pop(0)
        if depth + 1 >= max_depth:
            continue
        try:
            items = list(node._modules.items())
        except Exception:
            continue
        for k, v in items:
            seen += 1
            if v is child:
                return f"{prefix}{k}"
            if v is not None:
                frontier.append((f"{prefix}{k}.", v, depth + 1))
            if seen >= budget:
                break
    return None


def _path_frame(module: "torch.nn.Module") -> tuple[str, str]:
    """返回 (完整层级路径, 类名)；嵌套关系由本模块自己的调用栈推导。"""
    stack = getattr(_TLS, "paths", None)
    if stack is None:
        stack = []
        _TLS.paths = stack
    cls = type(module).__name__
    if stack:
        parent_path, parent_mod = stack[-1]
        path = f"{parent_path}.{_child_name(parent_mod, module)}" if parent_path else _child_name(parent_mod, module)
    else:
        path = cls
    stack.append((path, module))
    return path, cls


def _pop_path_frame() -> None:
    stack = getattr(_TLS, "paths", None)
    if stack:
        stack.pop()


def current_module_path() -> str:
    """供外部（如 functional hook）拼接调用上下文。

    栈里存的已经是「绝对路径」，所以只能取栈顶；把它们逐个 join 会得到
    ``TinyNet.TinyNet.fc1`` 这种重复前缀。
    """
    stack = getattr(_TLS, "paths", None)
    return stack[-1][0] if stack else ""


# ----------------------------------------------------------------------
# 安装
# ----------------------------------------------------------------------
def install(rec: Recorder) -> bool:
    cfg = rec.cfg
    if _STATE["installed"]:
        return False
    _STATE["rec"] = rec
    _STATE["cfg"] = cfg
    _silence_hook_warnings(cfg)
    orig = torch.nn.Module.__call__
    _ORIG["call"] = orig

    def patched_call(self, *args, **kwargs):
        rec_local = _STATE["rec"]
        if rec_local is None or not rec_local.active or _skip_module(self):
            return orig(self, *args, **kwargs)
        path, cls = _path_frame(self)
        try:
            return _forward(rec_local, self, path, cls, args, kwargs)
        except PProbeLimitReached:
            raise                         # 用户显式要求中断（PPROBE_STOP_ON_NAN），不能当作内部异常吞掉
        except Exception as e:  # 探针内部异常绝不允许打断训练
            rec_local.note_error(f"fwd:{type(e).__name__}")
            try:
                _safe_fallback_log(e, path)
            except Exception:
                pass
            return orig(self, *args, **kwargs)
        finally:
            _pop_path_frame()

    torch.nn.Module.__call__ = patched_call
    _STATE["orig_call"] = orig
    _STATE["installed"] = True
    _STATE["patched_call"] = patched_call

    if cfg.needs_backward():
        _install_backward_entry(rec, cfg)
    if cfg.has("func"):
        _patch_functional(rec, cfg)
    if cfg.has("optim"):
        _patch_optimizer(rec, cfg)
    if cfg.verbose:
        log(f"已安装 forward/backward hook（{cfg.banner()}）")
    return True


def uninstall() -> None:
    """还原所有被替换的函数（测试/自停用）。"""
    if _ORIG.get("call") is not None:
        torch.nn.Module.__call__ = _ORIG["call"]
    for key, spec in list(_ORIG.items()):
        if key == "call" or not isinstance(spec, tuple) or len(spec) != 3:
            continue
        owner, attr, orig_fn = spec
        try:
            setattr(owner, attr, orig_fn)
        except Exception:
            pass
    _ORIG.clear()
    _STATE.update({"installed": False, "rec": None, "cfg": None, "orig_call": None})


def _silence_hook_warnings(cfg) -> None:
    """探针自己引入的告警不该刷屏（``PPROBE_KEEP_WARNINGS=1`` 可保留）。"""
    if os.environ.get("PPROBE_KEEP_WARNINGS"):
        return
    import warnings

    for pattern in (
        "Full backward hook is firing when gradients are computed",
        "Forward hook registered with always_call",
    ):
        warnings.filterwarnings("ignore", message=pattern + "*")


def _safe_fallback_log(e: Exception, path: str) -> None:
    rec = _STATE["rec"]
    if rec is not None and rec.cfg.verbose and len(rec._errors) < 24:
        log(f"hook 异常已吞掉（module={path}）: {e!r}")


def _grad_enabled() -> bool:
    try:
        return torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    except Exception:
        return torch.is_grad_enabled()


def _should_skip_framework(self) -> bool:
    """跳过 dynamo/fx 追溯阶段，避免把编译期符号当成真实数值。"""
    try:
        if torch.jit.is_tracing():
            return True
    except Exception:
        pass
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
    return False


# ----------------------------------------------------------------------
# forward 记录
# ----------------------------------------------------------------------
def _forward(rec: Recorder, module: "torch.nn.Module", path: str, cls: str,
             args: tuple, kwargs: dict) -> Any:
    cfg = rec.cfg
    orig = _ORIG["call"]
    st = _mod_state(module, path, cls)

    want = rec.want_module(path, cls) and rec.budget_left(path)
    stack_id = rec.stacks.capture() if (cfg.stacks and want) else None
    call_index = rec.call_index(path) if want else st.call_index
    st.call_index = call_index + 1
    if want:
        st.rec_call_index = call_index

    if want and cfg.needs_backward() and cfg.bwd_mode in ("module", "both") and _grad_enabled() \
            and not _should_skip_framework(module):
        _prepare_module_backward(rec, module, st)

    out = orig(module, *args, **kwargs)
    if not want:
        return out

    try:
        if _should_skip_framework(module):
            return out
        collector = _make_collector(rec, path, "forward")
        collect_args(args, kwargs, collector)
        collector.add("output", out)
        if cfg.record_params and not st.params_done:
            collect_params(module, collector)
            st.params_done = True
        extra = {"output_sig": structure_sig(out, 1), "grad_enabled": _grad_enabled()}
        if collector.truncated:
            extra["truncated"] = True
        seq = rec.record("forward", path, cls, collector.slots, stack_id=stack_id,
                         call_index=call_index, extra=extra)
        st.fwd_seq = seq
        st.fwd_stack = stack_id
        # bwd_failed：模块级 hook 注册失败时自动回退到张量级 hook，保证反向不缺席
        want_tensor_bwd = cfg.bwd_mode in ("tensor", "both") or st.bwd_failed
        if want_tensor_bwd and not module.__dict__.get("_pprobe_tensor_bwd") and _grad_enabled():
            _register_tensor_grad_hooks(rec, module, args, kwargs, out, st)
        if cfg.stop_on_nan and has_nan(collector.slots):
            rec.record_nan(path, "forward")
    except Exception as e:
        rec.note_error(f"fwd-record:{type(e).__name__}")
        if isinstance(e, RecursionError) or cfg.stop_on_nan:
            raise
    return out


def _make_collector(rec: Recorder, module: str, phase: str) -> SlotCollector:
    cfg = rec.cfg
    saver = rec.make_full_saver(module, phase, 0)
    return SlotCollector(cfg, rec.sampler, full_saver=saver)


# ----------------------------------------------------------------------
# backward 记录
# ----------------------------------------------------------------------
def _install_backward_entry(rec: Recorder, cfg) -> None:
    """记录一次 .backward()/autograd.backward 调用的序号，便于把反向事件分组。"""
    key = "autograd_backward"
    if _ORIG.get(key) is not None:
        return
    orig = torch.autograd.backward

    def wrapper(*args, **kwargs):
        rec.bwd_index += 1
        return orig(*args, **kwargs)

    wrapper._pprobe_wrapped = True  # type: ignore[attr-defined]
    _ORIG[key] = (torch.autograd, "backward", orig)
    torch.autograd.backward = wrapper  # type: ignore[assignment]


def _prepare_module_backward(rec: Recorder, module: "torch.nn.Module", st: _ModState) -> None:
    """注册模块级 full backward hook（torch 要求在 forward 之前注册，故在调用前做）。"""
    if st.bwd_registered or st.bwd_failed:
        return
    try:
        ref = weakref.ref(module)

        def hook(mod, grad_input, grad_output):
            try:
                _record_module_backward(rec, mod, grad_input, grad_output, st)
            except Exception as e:
                rec.note_error(f"bwd:{type(e).__name__}")
                if rec.cfg.stop_on_nan:
                    raise
            return None  # 返回 None 表示不修改任何梯度

        module.register_full_backward_hook(hook)
        st.bwd_registered = True
        del ref
    except Exception as e:  # 无参数模块 / 特殊模块注册失败 -> 回退到 tensor hook
        st.bwd_failed = True
        rec.note_error(f"bwd-register:{type(e).__name__}")


def _record_module_backward(rec: Recorder, module: "torch.nn.Module", grad_input, grad_output,
                            st: _ModState) -> None:
    cfg = rec.cfg
    if not rec.active or not rec.want_module(st.path, st.cls) or not rec.budget_left(st.path):
        return
    collector = _make_collector(rec, st.path, "backward")
    for i, g in enumerate(grad_input or ()):
        collector.add(f"grad_input[{i}]", g)
    for i, g in enumerate(grad_output or ()):
        collector.add(f"grad_output[{i}]", g)
    stack_id = rec.stacks.capture() if cfg.stacks else None
    rec.record(
        "backward", st.path, st.cls, collector.slots, stack_id=stack_id,
        fwd_seq=st.fwd_seq, call_index=st.bwd_call_index(),
        extra={"bwd_src": "module-hook", "bwd_index": rec.bwd_index,
               "fwd_stack_id": st.fwd_stack},
    )
    if cfg.stop_on_nan and has_nan(collector.slots):
        rec.record_nan(st.path, "backward")


def _iter_boundary_tensors(args, kwargs, out):
    """只取模块边界上的张量（一层深度），避免为整棵嵌套结构注册梯度 hook。"""
    for i, v in enumerate(args):
        yield f"input[{i}]", v
    for k, v in (kwargs or {}).items():
        yield f"kwargs.{k}", v
    if isinstance(out, torch.Tensor):
        yield "output", out
    elif isinstance(out, (tuple, list)):
        for i, v in enumerate(out):
            if isinstance(v, torch.Tensor):
                yield f"output[{i}]", v
    elif isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                yield f"output.{k}", v


def _can_register_grad_hook(t: "torch.Tensor") -> bool:
    try:
        return bool(t.requires_grad) and (t.grad_fn is not None or t.is_leaf) and not t.is_sparse
    except Exception:
        return False


def _register_tensor_grad_hooks(rec: Recorder, module: "torch.nn.Module", args, kwargs, out,
                                st: _ModState) -> None:
    """Tensor.register_hook 兜底：拿到「对某个边界张量的梯度」。"""
    module.__dict__["_pprobe_tensor_bwd"] = True
    registered = 0
    for slot, t in _iter_boundary_tensors(args, kwargs, out):
        if not isinstance(t, torch.Tensor) or not _can_register_grad_hook(t):
            continue
        if registered >= 8:  # 每个模块边界张量最多 8 个，防止极端 fan-out
            break
        registered += 1
        _attach_grad_hook(rec, t, slot, st)


def _attach_grad_hook(rec: Recorder, t: "torch.Tensor", slot: str, st: _ModState) -> None:
    cfg = rec.cfg
    key = id(t)

    def hook(grad):
        try:
            if not rec.active or not rec.want_module(st.path, st.cls):
                return None
            occ = _GRAD_OCCURRENCE.get(key, 0)
            _GRAD_OCCURRENCE[key] = occ + 1
            if occ >= 4:  # 同一张量最多记录 4 次梯度，避免 diamond 依赖刷屏
                return None
            collector = _make_collector(rec, st.path, "backward")
            collector.add(f"grad_of:{slot}", grad)
            rec.record(
                "backward", st.path, st.cls, collector.slots,
                stack_id=rec.stacks.capture() if cfg.stacks else None,
                fwd_seq=st.fwd_seq, call_index=st.bwd_call_index(),
                extra={"bwd_src": "tensor-hook", "occurrence": occ, "bwd_index": rec.bwd_index},
            )
        except Exception as e:
            rec.note_error(f"grad-hook:{type(e).__name__}")
        return None  # 不改动梯度

    try:
        t.register_hook(hook)
    except Exception as e:
        rec.note_error(f"grad-register:{type(e).__name__}")


# ----------------------------------------------------------------------
# 函数级 hook（torch.nn.functional），用于定位没有对应 Module 的算子
# ----------------------------------------------------------------------
def _patch_functional(rec: Recorder, cfg) -> None:
    import torch.nn.functional as F

    from .config import DEFAULT_FUNC_TARGETS

    # 直接构造 Config（单测/嵌入使用）时 func_targets 可能为空，此时用内置默认名单，
    # 否则 ``hooks=(..., "func")`` 会静默什么都不 hook。
    for name in cfg.func_targets or DEFAULT_FUNC_TARGETS:
        fn = getattr(F, name, None)
        if fn is None or getattr(fn, "_pprobe_wrapped", False):
            continue
        if _ORIG.get(("func", name)) is None:
            _ORIG[("func", name)] = (F, name, fn)
        setattr(F, name, _make_func_wrapper(name, fn, rec, cfg))
        if cfg.verbose:
            log(f"已 hook torch.nn.functional.{name}")


def _make_func_wrapper(name: str, fn, rec: Recorder, cfg):
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not rec.active or _should_skip_framework(None):
            return fn(*args, **kwargs)
        idx = _FUNC_CALLS.get(name, 0)
        _FUNC_CALLS[name] = idx + 1
        out = fn(*args, **kwargs)
        try:
            path = f"functional.{name}"
            collector = _make_collector(rec, path, "forward")
            collect_args(args, kwargs, collector)
            collector.add("output", out)
            stack_id = rec.stacks.capture() if cfg.stacks else None
            seq = rec.record("forward", path, "Function", collector.slots, stack_id=stack_id,
                             call_index=idx, extra={"output_sig": structure_sig(out, 1),
                                                    "in_module": current_module_path()})
            if cfg.needs_backward() and _grad_enabled() and isinstance(out, torch.Tensor) and _can_register_grad_hook(out):
                st = _ModState(path, "Function")
                st.fwd_seq = seq
                st.call_index = idx
                st.rec_call_index = idx
                _attach_grad_hook(rec, out, "output", st)
        except Exception as e:
            rec.note_error(f"func-record:{type(e).__name__}")
        return out

    wrapper._pprobe_wrapped = True  # type: ignore[attr-defined]
    return wrapper


# ----------------------------------------------------------------------
# optimizer step 计数（用于给事件打上 step 标签）
# ----------------------------------------------------------------------
def _patch_optimizer(rec: Recorder, cfg) -> None:
    """统计优化器 step 次数（给事件打 ``step`` 标签，并支持 PPROBE_MAX_STEPS）。

    不能只包 ``Optimizer.step``：SGD/Adam 等子类都自己定义了 ``step``，先在自己类上
    查到方法，包基类根本不生效。改成包 ``Optimizer.__init__``，在实例化时给具体类包一层。
    """
    import functools

    import torch.optim as optim

    orig_init = optim.Optimizer.__init__
    if getattr(orig_init, "_pprobe_wrapped", False):
        return
    wrapped_classes: set[int] = set()

    def wrap_step_of(cls):
        owner = next((c for c in cls.__mro__ if "step" in c.__dict__), None)
        if owner is None or id(owner) in wrapped_classes:
            return
        wrapped_classes.add(id(owner))
        orig_step = owner.__dict__["step"]
        if getattr(orig_step, "_pprobe_wrapped", False):
            return

        @functools.wraps(orig_step)
        def step(self, *args, **kwargs):
            out = orig_step(self, *args, **kwargs)
            try:
                if rec.active and not getattr(self, "_pprobe_no_step_count", False):
                    rec.on_optimizer_step()
            except Exception as e:
                rec.note_error(f"optim:{type(e).__name__}")
            return out

        step._pprobe_wrapped = True  # type: ignore[attr-defined]
        try:
            setattr(owner, "step", step)
            # 必须登记原始 step：否则 uninstall 后子类上仍留着包旧 recorder 的壳，
            # 新 recorder 再装时被 ``_pprobe_wrapped`` 短路，step 计数从此永远不生效。
            _ORIG[("optim_step", id(owner))] = (owner, "step", orig_step)
        except Exception:
            pass

    @functools.wraps(orig_init)
    def new_init(self, *args, **kwargs):
        try:
            wrap_step_of(type(self))
        except Exception:
            pass
        return orig_init(self, *args, **kwargs)

    new_init._pprobe_wrapped = True  # type: ignore[attr-defined]
    _ORIG["optim_init"] = (optim.Optimizer, "__init__", orig_init)
    optim.Optimizer.__init__ = new_init  # type: ignore[assignment]
    _ORIG["optim_classes"] = wrapped_classes  # type: ignore[assignment]
