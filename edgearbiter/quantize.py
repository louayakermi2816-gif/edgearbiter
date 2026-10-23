"""Post-training static INT8 quantisation (torch.ao.quantization, FX graph mode).

Works on the plain torchvision architectures trained by `edgearbiter.finetune`
(MobileNetV3-Small has no eager-mode quantisable variant in torchvision, FX mode needs
none): observers are inserted, a few hundred training images calibrate the activation
ranges, and the graph is converted to INT8 kernels for the chosen engine ("x86" /
"fbgemm" on Intel/AMD, "qnnpack" on ARM). The INT8 accuracy is then measured like any
other model, so FP32 vs INT8 is compared on the same validation images.
"""
from __future__ import annotations

import copy
import warnings
from typing import Optional

import torch

from .benchmark import set_quant_engine


def quantize_ptq(model: torch.nn.Module, calib_loader, backend: Optional[str] = None, batches: int = 16,
                 img: int = 128) -> torch.nn.Module:
    """INT8 copy of `model` calibrated on `batches` batches of `calib_loader`."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    engine = set_quant_engine(backend)
    example = (torch.rand(1, 3, img, img),)
    with warnings.catch_warnings():                 # FX quantisation is deprecated upstream (torchao), still shipped
        warnings.filterwarnings("ignore", message=r".*(deprecated|quantized tensor creation functions)")
        prepared = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping(engine), example)
        with torch.no_grad():
            for b, (x, _) in enumerate(calib_loader):
                prepared(x)
                if b + 1 >= batches:
                    break
        quantized = convert_fx(prepared)
    quantized.quant_engine = engine
    return quantized.eval()


def load_quantized(model: torch.nn.Module, state_path, backend: Optional[str] = None, img: int = 128) -> torch.nn.Module:
    """INT8 model from a saved `quantize_ptq` state: FX graph modules do not pickle, so the INT8 graph is
    rebuilt (uncalibrated) from the FP32 `model` and the calibrated weights and scales are loaded into it."""
    quantized = quantize_ptq(model, [(torch.rand(1, 3, img, img), None)], backend=backend, batches=1, img=img)
    quantized.load_state_dict(torch.load(state_path, weights_only=False))
    return quantized
