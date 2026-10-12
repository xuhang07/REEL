"""Equivalent GEMM for Qwen's one-kernel-per-patch vision projection.

Keep the Conv3d parameters, state-dict names and autograd graph. Each input
patch has exactly the spatial/temporal dimensions of the convolution kernel,
so its single output voxel is a linear projection of C,T,H,W in that order.
"""
from types import MethodType

import torch.nn.functional as F


def linear_patch_forward(self, hidden_states):
    proj = self.proj
    # Preserve module hook semantics if a caller has instrumented the Conv3d.
    if (proj._forward_hooks or proj._forward_pre_hooks
            or proj._backward_hooks or proj._backward_pre_hooks):
        return self._reel_conv_patch_forward(hidden_states)
    width = self.in_channels * self.temporal_patch_size * self.patch_size**2
    patches = hidden_states.view(-1, width).to(dtype=proj.weight.dtype)
    return F.linear(patches, proj.weight.flatten(1), proj.bias)


def enable_linear_patch_embed(model):
    modules = [m for m in model.modules()
               if type(m).__name__ in {"Qwen3_5VisionPatchEmbed", "Qwen3_5MoeVisionPatchEmbed"}]
    # Validate all targets before mutating any of them.
    for module in modules:
        proj = module.proj
        kernel = (module.temporal_patch_size, module.patch_size, module.patch_size)
        if not (tuple(proj.kernel_size) == kernel and tuple(proj.stride) == kernel
                and tuple(proj.padding) == (0, 0, 0) and tuple(proj.dilation) == (1, 1, 1)
                and proj.groups == 1 and proj.padding_mode == "zeros"
                and proj.out_channels == module.embed_dim and proj.in_channels == module.in_channels):
            raise ValueError("Unsupported Qwen vision patch projection geometry")
    for module in modules:
        if not hasattr(module, "_reel_conv_patch_forward"):
            module._reel_conv_patch_forward = module.forward
            module.forward = MethodType(linear_patch_forward, module)
    return len(modules)


def disable_linear_patch_embed(model):
    count = 0
    for module in model.modules():
        if hasattr(module, "_reel_conv_patch_forward"):
            module.forward = module._reel_conv_patch_forward
            del module._reel_conv_patch_forward
            count += 1
    return count
