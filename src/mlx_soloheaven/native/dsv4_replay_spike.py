"""Stage 3a spike: run MLX's precompiled qmv from OUR OWN Metal command
buffer, on MLX-owned buffers obtained via DLPack — zero copy, zero MLX
involvement in the dispatch. Success criteria: output matches
mx.quantized_matmul bit-for-bit-ish (same kernel, same data)."""
import ctypes
import ctypes.util

import mlx.core as mx
import numpy as np

# --- objc runtime plumbing --------------------------------------------------
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]
objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]


def sel(name: bytes):
    return objc.sel_registerName(name)


def msg(obj, selector: bytes, *args, restype=ctypes.c_void_p, argtypes=()):
    send = objc.objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return send(obj, sel(selector), *args)


class MTLSize(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint64), ("h", ctypes.c_uint64), ("d", ctypes.c_uint64)]


def nsstring(s: str):
    return msg(objc.objc_getClass(b"NSString"), b"stringWithUTF8String:",
               s.encode(), argtypes=[ctypes.c_char_p])


# --- DLPack -> MTLBuffer ----------------------------------------------------
class DLDevice(ctypes.Structure):
    _fields_ = [("t", ctypes.c_int32), ("i", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("ctx", ctypes.c_void_p), ("del_", ctypes.c_void_p)]


_keep = []  # keep capsules alive


def mtl_buffer(a: mx.array):
    cap = a.__dlpack__()
    _keep.append(cap)
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    mt = ctypes.cast(
        ctypes.pythonapi.PyCapsule_GetPointer(cap, b"dltensor"),
        ctypes.POINTER(DLManagedTensor),
    ).contents
    assert mt.dl_tensor.device.t == 8, "expected kDLMetal"
    return mt.dl_tensor.data, mt.dl_tensor.byte_offset


# --- the experiment ---------------------------------------------------------
N, K, GS, BITS = 2048, 4096, 64, 8
w = mx.random.normal((N, K)).astype(mx.bfloat16)  # bf16 scales, like the real build
qw, sc, bi = mx.quantize(w, group_size=GS, bits=BITS)
x = mx.random.normal((1, K)).astype(mx.bfloat16)
y_ref = mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=GS, bits=BITS)
y_ours = mx.zeros((1, N), dtype=mx.bfloat16)
mx.eval(qw, sc, bi, x, y_ref, y_ours)
mx.synchronize()  # everything MLX wrote is committed before we touch buffers

dev = ctypes.CDLL(None).MTLCreateSystemDefaultDevice
dev.restype = ctypes.c_void_p
device = dev()
print("device ok:", bool(device))

lib_path = ".venv/lib/python3.14/site-packages/mlx/lib/mlx.metallib"
url = msg(objc.objc_getClass(b"NSURL"), b"fileURLWithPath:", nsstring(lib_path),
          argtypes=[ctypes.c_void_p])
err = ctypes.c_void_p(0)
lib = msg(device, b"newLibraryWithURL:error:", url, ctypes.byref(err),
          argtypes=[ctypes.c_void_p, ctypes.c_void_p])
print("metallib loaded:", bool(lib))

fn = msg(lib, b"newFunctionWithName:",
         nsstring("affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0"),
         argtypes=[ctypes.c_void_p])
print("function:", bool(fn))
pso = msg(device, b"newComputePipelineStateWithFunction:error:", fn, ctypes.byref(err),
          argtypes=[ctypes.c_void_p, ctypes.c_void_p])
print("pipeline:", bool(pso))

queue = msg(device, b"newCommandQueue")
cmd = msg(queue, b"commandBuffer")
enc = msg(cmd, b"computeCommandEncoder")
msg(enc, b"setComputePipelineState:", pso, argtypes=[ctypes.c_void_p])

bufs = [mtl_buffer(a) for a in (qw, sc, bi, x, y_ours)]
for i, (b, off) in enumerate(bufs):
    msg(enc, b"setBuffer:offset:atIndex:", b, off, i,
        argtypes=[ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64])
kv = ctypes.c_int32(K)
nv = ctypes.c_int32(N)
msg(enc, b"setBytes:length:atIndex:", ctypes.byref(kv), 4, 5,
    argtypes=[ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64])
msg(enc, b"setBytes:length:atIndex:", ctypes.byref(nv), 4, 6,
    argtypes=[ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64])

grid = MTLSize(1, (N + 7) // 8, 1)
group = MTLSize(32, 2, 1)
msg(enc, b"dispatchThreadgroups:threadsPerThreadgroup:", grid, group,
    argtypes=[MTLSize, MTLSize])
msg(enc, b"endEncoding")
msg(cmd, b"commit")
msg(cmd, b"waitUntilCompleted")
status = msg(cmd, b"status", restype=ctypes.c_int64)
print("command buffer status:", status, "(4=completed)")

a = np.array(y_ours.astype(mx.float32))
b = np.array(y_ref.astype(mx.float32))
print("max abs diff vs mx.quantized_matmul:", float(np.abs(a - b).max()))
print("allclose:", bool(np.allclose(a, b, atol=1e-2)))
