"""Keep invalid network values from being concealed by output quantization."""
import torch


def finite_output(value,expected_shape):
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(expected_shape) or not torch.isfinite(value).all():
        raise ValueError('Network output shape/nonfinite validation failed before clipping')
    return value
