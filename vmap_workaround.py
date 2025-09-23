# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

from functools import wraps
from typing import Any, Sequence, Callable
import torch
from torch.overrides import TorchFunctionMode

# This module defines a helper class and wrapper for working around the torch.vmap issue documented in
# https://github.com/pytorch/pytorch/issues/124423. This workaround was initially taken from
# https://github.com/EMI-Group/evox/blob/edb94acf2bcae2d054c2ada455d30504976b85cc/src/evox/core/module.py#L154
# and built out further to handle additional edge cases. This module should ultimately provide a direct 
# replacement for torch.vmap:

def _transform_scalar_index(ori_index: Sequence[Any | torch.Tensor] | Any | torch.Tensor):
    if isinstance(ori_index, Sequence):
        index = tuple(ori_index)
    else:
        index = (ori_index,)
    any_scalar_tensor = False
    new_index = []
    for idx in index:
        if isinstance(idx, torch.Tensor) and idx.ndim == 0:
            new_index.append(idx[None])
            any_scalar_tensor = True
        elif isinstance(idx, slice) and isinstance(idx.stop, torch.Tensor) and idx.stop.ndim == 0:  # NOTE: found really weird case where the underlying input tensor is buried in a slice
            new_index.append(idx.stop[None])
            any_scalar_tensor = True
        else:
            new_index.append(idx)
    if not isinstance(ori_index, Sequence):
        new_index = new_index[0]
    if type(new_index) == type([]):
        new_index = tuple(new_index)    # NOTE: added conversion from List to Tuple here to avoid UserWarning about using a non-tuple sequence for multidimensional indexing
    return new_index, any_scalar_tensor  

class TransformGetSetItemToIndex(TorchFunctionMode):
    # This is needed since we want to support calling
    # A[idx] or A[idx] += b, where idx is a scalar tensor.
    # When idx is a scalar tensor, Torch implicitly convert it to a python
    # scalar and create a view of A.
    # Workaround: We convert the scalar tensor to a 1D tensor with one element.
    # That is, we convert A[idx] to A[idx[None]][0], A[idx] += 1 to A[idx[None]] += 1.
    # This is a temporary solution until the issue is fixed in PyTorch.
    def __torch_function__(self, func, types, args, kwargs=None):
        # A[idx]
        if func == torch.Tensor.__getitem__:
            x, index = args
            new_index, any_scalar = _transform_scalar_index(index)
            x = func(x, new_index, **(kwargs or {}))
            if any_scalar:
                x = x.squeeze(0)
            return x
        # A[idx] = value
        elif func == torch.Tensor.__setitem__:
            x, index, value = args
            new_index, _ = _transform_scalar_index(index)
            return func(x, new_index, value, **(kwargs or {}))
        return func(*args, **(kwargs or {}))
    
@wraps(torch.vmap)
def vmap(*args, **kwargs) -> Callable:
    # Fixes the `torch.vmap`'s issue with __getitem__ and __setitem__.
    # Related issue: https://github.com/pytorch/pytorch/issues/124423.
    vmapped = torch.vmap(*args, **kwargs)
    def wrapper(*args, **kwargs):
        with TransformGetSetItemToIndex():
            return vmapped(*args, **kwargs)
    return wrapper