"""dsv4 native decode runtime (Stage 3b) — Metal plumbing, plan builder, and
the C encode-loop binding.

This module owns the low-level pieces the replay loop needs; the per-layer
decode plan and its numerical verification are built on top of it (ladder in
README.md). Everything here is proven working by the Stage 3a spike; this
is that spike, generalized and made reusable.

Nothing in this module is imported by the serving path unless
SOLOHEAVEN_DSV4_NATIVE=1 selects it — it is inert otherwise.
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
_METALLIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
    ".venv/lib/python3.14/site-packages/mlx/lib/mlx.metallib",
)

# --- objc runtime -----------------------------------------------------------
_objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.sel_registerName.argtypes = [ctypes.c_char_p]
_objc.objc_getClass.restype = ctypes.c_void_p
_objc.objc_getClass.argtypes = [ctypes.c_char_p]


class MTLSize(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint64), ("h", ctypes.c_uint64), ("d", ctypes.c_uint64)]


def _msg(obj, selector: bytes, *args, restype=ctypes.c_void_p, argtypes=()):
    send = _objc.objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return send(obj, _objc.sel_registerName(selector), *args)


def _nsstring(s: str):
    return _msg(_objc.objc_getClass(b"NSString"), b"stringWithUTF8String:",
                s.encode(), argtypes=[ctypes.c_char_p])


# --- DLPack -> MTLBuffer ----------------------------------------------------
class _DLDevice(ctypes.Structure):
    _fields_ = [("t", ctypes.c_int32), ("i", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManaged(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("ctx", ctypes.c_void_p), ("del_", ctypes.c_void_p)]


def mtl_buffer(a: mx.array) -> tuple[int, int, object]:
    """(MTLBuffer ptr, byte_offset, capsule-to-keep-alive) for an MLX array.
    The capsule owns the DLPack view; it MUST stay referenced for the buffer
    pointer to remain valid, so callers store it for the session."""
    cap = a.__dlpack__()
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    mt = ctypes.cast(
        ctypes.pythonapi.PyCapsule_GetPointer(cap, b"dltensor"),
        ctypes.POINTER(_DLManaged),
    ).contents
    if mt.dl_tensor.device.t != 8:
        raise RuntimeError("expected a Metal (kDLMetal) array")
    return mt.dl_tensor.data, mt.dl_tensor.byte_offset, cap


# --- C encode loop ----------------------------------------------------------
_DYLIB = os.path.join(_HERE, "libdsv4enc.dylib")


def _ensure_dylib() -> ctypes.CDLL:
    src = os.path.join(_HERE, "encoder.m")
    if not os.path.exists(_DYLIB) or os.path.getmtime(_DYLIB) < os.path.getmtime(src):
        subprocess.run(
            ["clang", "-fobjc-arc", "-O2", "-shared",
             "-framework", "Metal", "-framework", "Foundation",
             src, "-o", _DYLIB],
            check=True,
        )
    return ctypes.CDLL(_DYLIB)


class _PlanItem(ctypes.Structure):
    _MAXB, _MAXBY = 16, 4
    _fields_ = [
        ("pso", ctypes.c_int32),
        ("n_bufs", ctypes.c_int32),
        ("buf_ids", ctypes.c_int32 * _MAXB),
        ("buf_offs", ctypes.c_uint64 * _MAXB),
        ("buf_slots", ctypes.c_int32 * _MAXB),
        ("n_bytes", ctypes.c_int32),
        ("bytes_off", ctypes.c_int32 * _MAXBY),
        ("bytes_len", ctypes.c_int32 * _MAXBY),
        ("bytes_slot", ctypes.c_int32 * _MAXBY),
        ("grid", ctypes.c_uint64 * 3),
        ("group", ctypes.c_uint64 * 3),
    ]


class Runtime:
    """Owns the Metal device/queue, the pipeline cache, and the C loop."""

    def __init__(self):
        create = ctypes.CDLL(None).MTLCreateSystemDefaultDevice
        create.restype = ctypes.c_void_p
        self.device = create()
        if not self.device:
            raise RuntimeError("no Metal device")
        self.queue = _msg(self.device, b"newCommandQueue")
        self._lib_metal = self._load_library_url(_METALLIB)
        self._lib_custom = None
        self._pipelines: dict[str, int] = {}
        self._psos: list[int] = []
        self._dylib = _ensure_dylib()
        self._dylib.dsv4_encode_commit.restype = ctypes.c_int
        self._dylib.dsv4_encode_commit.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p, ctypes.c_int,
        ]
        self._dylib.dsv4_buffer_contents.restype = ctypes.c_void_p
        self._dylib.dsv4_buffer_contents.argtypes = [ctypes.c_void_p]

    def _load_library_url(self, path: str):
        url = _msg(_objc.objc_getClass(b"NSURL"), b"fileURLWithPath:",
                   _nsstring(path), argtypes=[ctypes.c_void_p])
        err = ctypes.c_void_p(0)
        lib = _msg(self.device, b"newLibraryWithURL:error:", url,
                   ctypes.byref(err), argtypes=[ctypes.c_void_p, ctypes.c_void_p])
        if not lib:
            raise RuntimeError(f"failed to load metallib: {path}")
        return lib

    def load_custom_source(self, source: str) -> None:
        """Compile our own kernels (explicit-signature .metal) into a library."""
        err = ctypes.c_void_p(0)
        opts = _msg(_objc.objc_getClass(b"MTLCompileOptions"), b"new")
        self._lib_custom = _msg(
            self.device, b"newLibraryWithSource:options:error:",
            _nsstring(source), opts, ctypes.byref(err),
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )
        if not self._lib_custom:
            raise RuntimeError("custom kernel library failed to compile")

    def pipeline(self, name: str, custom: bool = False) -> int:
        """Index of the compute pipeline for `name` (cached)."""
        if name in self._pipelines:
            return self._pipelines[name]
        lib = self._lib_custom if custom else self._lib_metal
        fn = _msg(lib, b"newFunctionWithName:", _nsstring(name),
                  argtypes=[ctypes.c_void_p])
        if not fn:
            raise RuntimeError(f"kernel not found: {name}")
        err = ctypes.c_void_p(0)
        pso = _msg(self.device, b"newComputePipelineStateWithFunction:error:",
                   fn, ctypes.byref(err),
                   argtypes=[ctypes.c_void_p, ctypes.c_void_p])
        if not pso:
            raise RuntimeError(f"pipeline creation failed: {name}")
        idx = len(self._psos)
        self._psos.append(pso)
        self._pipelines[name] = idx
        return idx

    def commit(self, items: list[_PlanItem], bufs: list[int],
               const_blob: bytes, wait: bool = True) -> None:
        """Encode `items` against buffer table `bufs` and commit. Per-token
        varying values must already be written into the uniform buffer(s) that
        `bufs` references; `const_blob` holds only plan-static setBytes data."""
        arr = (_PlanItem * len(items))(*items)
        psos = (ctypes.c_void_p * len(self._psos))(*self._psos)
        buf_arr = (ctypes.c_void_p * len(bufs))(*bufs)
        rc = self._dylib.dsv4_encode_commit(
            self.queue, arr, len(items), psos, buf_arr, const_blob, 1 if wait else 0
        )
        if rc != 0:
            raise RuntimeError(f"encode/commit failed (rc={rc})")

    def buffer_contents(self, buf_ptr: int) -> int:
        return self._dylib.dsv4_buffer_contents(buf_ptr)


def plan_qmv(rt: Runtime, buf_ids: tuple, K: int, N: int,
             const_off_K: int, const_off_N: int) -> _PlanItem:
    """A single affine_qmv_fast dispatch item. buf_ids = (w, scales, biases,
    x, y) session-table indices; K/N int32 live in the const blob at the given
    offsets (bound at slots 5 and 6, per mlx v0.32 quantized.cpp)."""
    it = _PlanItem()
    it.pso = rt.pipeline("affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0")
    it.n_bufs = 5
    for i, (bid, slot) in enumerate(zip(buf_ids, range(5))):
        it.buf_ids[i] = bid
        it.buf_offs[i] = 0
        it.buf_slots[i] = slot
    it.n_bytes = 2
    it.bytes_off[0], it.bytes_len[0], it.bytes_slot[0] = const_off_K, 4, 5
    it.bytes_off[1], it.bytes_len[1], it.bytes_slot[1] = const_off_N, 4, 6
    it.grid[:] = [1, (N + 7) // 8, 1]
    it.group[:] = [32, 2, 1]
    return it
