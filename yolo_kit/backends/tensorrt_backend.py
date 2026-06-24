from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class TensorRTBackendConfig:
    """
    Configuration for TensorRT engine inference.

    Notes:
    - TensorRT engines require a CUDA-capable environment.
    - This backend uses TensorRT plus CUDA runtime bindings directly.
    """

    device: str = "cuda"
    input_name: Optional[str] = None
    output_name: Optional[str] = None
    output_index: int = 0


@dataclass
class _BindingState:
    name: str
    index: int
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    nbytes: int
    device_ptr: int
    host_array: Optional[np.ndarray] = None


def _try_import_tensorrt():
    try:
        import tensorrt as trt
    except Exception as exc:
        raise ImportError(
            "tensorrt library is requried for the TensorRT backend, install NVIDIA TensorRT python bindings to proceed"
        ) from exc
    return trt


def _try_import_cudart():
    try:
        from cuda.bindings import runtime as cudart  # type: ignore

        return cudart
    except Exception:
        try:
            from cuda import cudart  # type: ignore

            return cudart
        except Exception as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "TensorRT backend requires CUDA runtime bindings. "
                "Install a compatible `cuda-python` package on the target device."
            ) from exc


def _cuda_call(cudart, result):
    if not isinstance(result, tuple):
        raise TypeError(f"Unexpected CUDA runtime return type: {type(result)}")
    if len(result) == 0:
        raise TypeError("Unexpected CUDA runtime return tuple: empty")
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        name = getattr(err, "name", None) or str(err)
        raise RuntimeError(f"CUDA runtime call failed: {name}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _parse_cuda_device_id(device: str) -> int:
    raw = str(device or "").strip().lower()
    if raw in {"cuda", "gpu"}:
        return 0
    if raw.startswith("cuda:"):
        suffix = raw.split(":", 1)[1].strip()
        if suffix.isdigit():
            return int(suffix)
    raise ValueError("TensorRTBackend requires a CUDA device string like 'cuda' or 'cuda:0'.")


def _trt_dtype_to_numpy_dtype(trt, trt_dtype) -> np.dtype:
    if hasattr(trt, "nptype"):
        try:
            return np.dtype(trt.nptype(trt_dtype))
        except Exception:
            pass

    name = getattr(trt_dtype, "name", str(trt_dtype)).lower()
    if "float16" in name or "fp16" in name or name == "half":
        return np.dtype(np.float16)
    if "float32" in name or "fp32" in name or name == "float":
        return np.dtype(np.float32)
    if "int8" in name:
        return np.dtype(np.int8)
    if "int32" in name:
        return np.dtype(np.int32)
    if "bool" in name:
        return np.dtype(np.bool_)
    raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype!r}")


class TensorRTBackend:
    """
    Minimal TensorRT engine runner without a torch dependency.

    Supports both classic binding-based execution (`execute_async_v2`) and the
    newer tensor-name API (`set_tensor_address` + `execute_async_v3`) depending
    on what the installed TensorRT version exposes. Please ennsure your engine
    is built in a way that is compatible with the API available in your environment.
    """

    def __init__(self, engine_path: PathLike, cfg: TensorRTBackendConfig = TensorRTBackendConfig()):
        trt = _try_import_tensorrt()
        cudart = _try_import_cudart()

        self._trt = trt
        self._cudart = cudart
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found in : {self.engine_path}")

        self.device_id = _parse_cuda_device_id(cfg.device)
        _cuda_call(cudart, cudart.cudaSetDevice(int(self.device_id)))

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = self.engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        stream = _cuda_call(cudart, cudart.cudaStreamCreate())
        self._logger = logger
        self._runtime = runtime
        self.engine = engine
        self.context = context
        self._stream = int(stream)
        self._use_io_tensors = hasattr(engine, "num_io_tensors") and hasattr(context, "set_tensor_address")

        if hasattr(self.context, "set_optimization_profile_async"):
            try:
                self.context.set_optimization_profile_async(0, self._stream)
            except Exception:
                pass

        self._bindings_meta = self._get_bindings_meta()
        self.input_name, self.output_names = self._discover_io(cfg.input_name)
        if cfg.output_name is not None:
            if cfg.output_name not in self.output_names:
                raise ValueError(f"Output name {cfg.output_name!r} not found. Available: {self.output_names}")
            self.primary_output = cfg.output_name
        else:
            if cfg.output_index < 0 or cfg.output_index >= len(self.output_names):
                raise IndexError(f"output_index {cfg.output_index} out of range (num outputs={len(self.output_names)}).")
            self.primary_output = self.output_names[cfg.output_index]

        self._binding_states: Dict[str, _BindingState] = {}
        self._bindings_ptrs: Optional[List[int]] = None
        self._last_input_shape: Optional[Tuple[int, ...]] = None

    def _get_bindings_meta(self) -> Sequence[Tuple[int, str]]:
        if hasattr(self.engine, "num_bindings") and hasattr(self.engine, "get_binding_name"):
            n = int(self.engine.num_bindings)
            return [(i, str(self.engine.get_binding_name(i))) for i in range(n)]

        if hasattr(self.engine, "num_io_tensors") and hasattr(self.engine, "get_tensor_name"):
            n = int(self.engine.num_io_tensors)
            names = [str(self.engine.get_tensor_name(i)) for i in range(n)]
            if hasattr(self.engine, "get_binding_index"):
                try:
                    return [(int(self.engine.get_binding_index(name)), name) for name in names]
                except Exception:
                    pass
            if not self._use_io_tensors:
                raise RuntimeError(
                    "TensorRT engine exposes IO tensors but no binding indices, and v3 execution is unavailable."
                )
            return list(enumerate(names))

        raise RuntimeError("Unsupported TensorRT engine API (missing IO/binding enumeration)")

    def _tensor_is_input(self, name: str, index: int) -> bool:
        trt = self._trt
        if hasattr(self.engine, "get_tensor_mode") and hasattr(trt, "TensorIOMode"):
            return bool(self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        if hasattr(self.engine, "binding_is_input"):
            return bool(self.engine.binding_is_input(index))
        raise RuntimeError("Unsupported TensorRT engine API (cannot query input/output)")

    def _tensor_dtype(self, name: str, index: int) -> np.dtype:
        if hasattr(self.engine, "get_tensor_dtype"):
            return _trt_dtype_to_numpy_dtype(self._trt, self.engine.get_tensor_dtype(name))
        if hasattr(self.engine, "get_binding_dtype"):
            return _trt_dtype_to_numpy_dtype(self._trt, self.engine.get_binding_dtype(index))
        raise RuntimeError("Unsupported TensorRT engine API (cannot query tensor dtype)")

    def _tensor_shape(self, name: str, index: int) -> Tuple[int, ...]:
        if hasattr(self.engine, "get_tensor_shape"):
            return tuple(int(x) for x in self.engine.get_tensor_shape(name))
        if hasattr(self.engine, "get_binding_shape"):
            return tuple(int(x) for x in self.engine.get_binding_shape(index))
        raise RuntimeError("Unsupported TensorRT engine API (cannot query tensor shape)")

    def _runtime_tensor_shape(self, name: str, index: int) -> Tuple[int, ...]:
        if hasattr(self.context, "get_tensor_shape"):
            return tuple(int(x) for x in self.context.get_tensor_shape(name))
        if hasattr(self.context, "get_binding_shape"):
            return tuple(int(x) for x in self.context.get_binding_shape(index))
        return self._tensor_shape(name, index)

    def _set_input_shape(self, name: str, index: int, shape: Tuple[int, ...]) -> None:
        current = self._tensor_shape(name, index)
        if not any(dim <= 0 for dim in current):
            if tuple(int(dim) for dim in current) != tuple(int(dim) for dim in shape):
                raise ValueError(f"Input shape mismatch: got {shape}, expected {current}")
            return
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(name, shape)
            return
        if hasattr(self.context, "set_binding_shape"):
            self.context.set_binding_shape(index, shape)
            return
        raise RuntimeError("Unsupported TensorRT context API (cannot set dynamic input shapes)")

    def _discover_io(self, preferred_input: Optional[str]) -> Tuple[str, List[str]]:
        inputs: List[str] = []
        outputs: List[str] = []
        for index, name in self._bindings_meta:
            if self._tensor_is_input(name, index):
                inputs.append(name)
            else:
                outputs.append(name)
        if not inputs:
            raise RuntimeError("TensorRT engine has no inputs.")
        if not outputs:
            raise RuntimeError("TensorRT engine has no outputs.")
        input_name = preferred_input or inputs[0]
        if input_name not in inputs:
            raise ValueError(f"Input name {input_name!r} not found. Available: {inputs}")
        return input_name, outputs

    def _free_device_ptr(self, device_ptr: int) -> None:
        if not device_ptr:
            return
        try:
            _cuda_call(self._cudart, self._cudart.cudaFree(int(device_ptr)))
        except Exception:
            pass

    def _build_bindings_ptrs(self) -> Optional[List[int]]:
        if self._use_io_tensors:
            return None
        if hasattr(self.engine, "num_bindings"):
            n = int(self.engine.num_bindings)
            ptrs = [0] * n
            for state in self._binding_states.values():
                if 0 <= state.index < n:
                    ptrs[state.index] = int(state.device_ptr)
            return ptrs
        return [int(state.device_ptr) for state in sorted(self._binding_states.values(), key=lambda item: item.index)]

    def _allocate_or_reuse_binding(self, *, name: str, index: int, is_input: bool) -> _BindingState:
        dtype = self._tensor_dtype(name, index)
        shape = self._runtime_tensor_shape(name, index)
        if any(dim <= 0 for dim in shape):
            raise RuntimeError(
                f"TensorRT tensor '{name}' has unresolved shape {shape}. "
                "Build a fixed-shape engine or ensure the runner sets input shapes correctly."
            )
        nbytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
        existing = self._binding_states.get(name)
        if (
            existing is not None
            and existing.shape == shape
            and existing.dtype == dtype
            and existing.nbytes == nbytes
            and int(existing.device_ptr) != 0
        ):
            return existing

        if existing is not None:
            self._free_device_ptr(existing.device_ptr)

        device_ptr = int(_cuda_call(self._cudart, self._cudart.cudaMalloc(nbytes)))
        host_array = None
        if (not is_input) and name == self.primary_output:
            host_array = np.empty(shape, dtype=dtype)
        state = _BindingState(
            name=name,
            index=index,
            is_input=is_input,
            dtype=dtype,
            shape=shape,
            nbytes=nbytes,
            device_ptr=device_ptr,
            host_array=host_array,
        )
        self._binding_states[name] = state
        return state

    def _ensure_bindings_ready(self, input_shape: Tuple[int, ...]) -> None:
        input_index = next(index for index, name in self._bindings_meta if name == self.input_name)
        self._set_input_shape(self.input_name, input_index, input_shape)

        if hasattr(self.context, "all_binding_shapes_specified"):
            try:
                if not self.context.all_binding_shapes_specified:
                    raise RuntimeError("TensorRT context shapes are not fully specified after setting input shapes.")
            except Exception:
                pass

        for index, name in self._bindings_meta:
            is_input = self._tensor_is_input(name, index)
            self._allocate_or_reuse_binding(name=name, index=index, is_input=is_input)

        self._bindings_ptrs = self._build_bindings_ptrs()
        self._last_input_shape = tuple(int(x) for x in input_shape)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        if blob is None:
            raise TypeError("blob must be a NumPy array.")

        arr = np.asarray(blob)
        if arr.ndim < 1:
            raise ValueError(f"Expected array-like input, got shape {arr.shape}")
        input_shape = tuple(int(x) for x in arr.shape)

        if self._last_input_shape != input_shape:
            self._ensure_bindings_ready(input_shape)

        input_state = self._binding_states[self.input_name]
        if arr.dtype != input_state.dtype:
            arr = arr.astype(input_state.dtype, copy=False)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if tuple(int(x) for x in arr.shape) != input_state.shape:
            raise ValueError(f"Input shape mismatch: got {arr.shape}, expected {input_state.shape}")

        cudart = self._cudart
        _cuda_call(
            cudart,
            cudart.cudaMemcpyAsync(
                int(input_state.device_ptr),
                int(arr.ctypes.data),
                int(input_state.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            ),
        )

        if self._use_io_tensors:
            for state in self._binding_states.values():
                self.context.set_tensor_address(state.name, int(state.device_ptr))
            ok = bool(self.context.execute_async_v3(self._stream))
        elif hasattr(self.context, "execute_async_v2"):
            assert self._bindings_ptrs is not None
            ok = bool(self.context.execute_async_v2(self._bindings_ptrs, self._stream))
        elif hasattr(self.context, "execute_v2"):
            assert self._bindings_ptrs is not None
            ok = bool(self.context.execute_v2(self._bindings_ptrs))
        else:  # pragma: no cover - defensive
            raise RuntimeError("Unsupported TensorRT execution context API (missing execute methods)")

        if not ok:
            raise RuntimeError("TensorRT execution failed")

        out_state = self._binding_states[self.primary_output]
        assert out_state.host_array is not None
        _cuda_call(
            cudart,
            cudart.cudaMemcpyAsync(
                int(out_state.host_array.ctypes.data),
                int(out_state.device_ptr),
                int(out_state.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            ),
        )

        _cuda_call(cudart, cudart.cudaStreamSynchronize(self._stream))
        return out_state.host_array

    def close(self) -> None:
        for state in list(self._binding_states.values()):
            self._free_device_ptr(state.device_ptr)
        self._binding_states.clear()
        try:
            if self._stream:
                _cuda_call(self._cudart, self._cudart.cudaStreamDestroy(self._stream))
        except Exception:
            pass
        self._stream = 0

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
