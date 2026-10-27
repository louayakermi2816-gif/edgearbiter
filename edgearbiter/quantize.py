"""Post-training static INT8 quantisation (torch.ao.quantization, FX graph mode).

Works on the plain torchvision architectures trained by `edgearbiter.finetune`
(MobileNetV3-Small has no eager-mode quantisable variant in torchvision, FX mode needs
none): observers are inserted, a few hundred training images calibrate the activation
ranges, and the graph is converted to INT8 kernels for the chosen engine ("x86" /
"fbgemm" on Intel/AMD, "qnnpack" on ARM). The INT8 accuracy is then measured like any
other model, so FP32 vs INT8 is compared on the same validation images.

Detectors (torchvision SSD / SSDLite): the backbone and the heads (`DetectorTrunk`) run in
INT8; resizing, anchors, box decoding and non-maximum suppression stay the FP32 model's own
(`QuantizedSSD`), so the FP32 and INT8 detectors differ only in the trunk.
"""
from __future__ import annotations

import copy
import warnings
from typing import Callable, Optional

import torch

from .benchmark import set_quant_engine


class DetectorTrunk(torch.nn.Module):
    """Backbone and heads of a torchvision SSD: the part of the detector that runs in INT8."""

    def __init__(self, ssd):
        super().__init__()
        self.backbone, self.head = ssd.backbone, ssd.head

    def forward(self, x):
        out = self.head(list(self.backbone(x).values()))
        return out["bbox_regression"], out["cls_logits"]


class QuantizedSSD(torch.nn.Module):
    """A torchvision SSD with an INT8 `DetectorTrunk`; same input and output as the SSD (a list of images in,
    a list of {"boxes", "scores", "labels"} out), same pre- and post-processing and thresholds."""

    def __init__(self, ssd, trunk: torch.nn.Module, example: torch.Tensor):
        super().__init__()
        self.transform, self.anchor_generator, self.box_coder, self.trunk = ssd.transform, ssd.anchor_generator, ssd.box_coder, trunk
        self.score_thresh, self.topk_candidates = ssd.score_thresh, ssd.topk_candidates
        self.nms_thresh, self.detections_per_img = ssd.nms_thresh, ssd.detections_per_img
        with torch.no_grad():                    # feature-map sizes for the anchors (the SSD resizes to a fixed size)
            self._grids = [torch.empty(0, 0, *f.shape[-2:]) for f in ssd.backbone(example).values()]

    def postprocess_detections(self, head_outputs, anchors, image_sizes):
        from torchvision.models.detection.ssd import SSD
        return SSD.postprocess_detections(self, head_outputs, anchors, image_sizes)

    def forward(self, images):
        sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        batch, _ = self.transform(images)
        bbox_regression, cls_logits = self.trunk(batch.tensors)
        detections = self.postprocess_detections({"bbox_regression": bbox_regression, "cls_logits": cls_logits},
                                                 self.anchor_generator(batch, self._grids), batch.image_sizes)
        return self.transform.postprocess(detections, batch.image_sizes, sizes)


def _ssd_input(ssd, x: torch.Tensor) -> torch.Tensor:
    """What an SSD's backbone sees for a batch in the loaders' format (ImageNet-normalised): the SSD resizes
    and normalises [0, 1] images itself."""
    from .data import MEAN, STD
    raw = (x * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)).clamp(0, 1)
    return ssd.transform(list(raw))[0].tensors


def detector_trunk(model: torch.nn.Module, img: int):
    """The trunk of an FP32 SSD or a `QuantizedSSD`, and an input for it: what INT8 quantisation changes."""
    x = model.transform([torch.rand(3, img, img)])[0].tensors
    return (model.trunk if isinstance(model, QuantizedSSD) else DetectorTrunk(model)), x


def _qconfig_mapping(engine: str, fast: bool = False):
    """The engine's default mapping; `fast`: min/max activation observers instead of histograms, same INT8
    graph and state keys (converting a detector takes ~0.5 s instead of ~2 min)."""
    from torch.ao.quantization import MinMaxObserver, QConfig, get_default_qconfig_mapping
    mapping = get_default_qconfig_mapping(engine)
    if fast:
        default = mapping.global_qconfig
        mapping.set_global(QConfig(activation=MinMaxObserver.with_args(reduce_range=default.activation().reduce_range),
                                   weight=default.weight))
    return mapping


def quantize_ptq(model: torch.nn.Module, calib_loader, backend: Optional[str] = None, batches: int = 16,
                 img: int = 128, fast: bool = False) -> torch.nn.Module:
    """INT8 copy of `model` calibrated on `batches` batches of `calib_loader` (a torchvision SSD: `QuantizedSSD`).
    `fast` observers (see `_qconfig_mapping`) for graphs whose calibrated state is loaded afterwards."""
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from torchvision.models.detection.ssd import SSD

    engine = set_quant_engine(backend)
    ssd = model.eval() if isinstance(model, SSD) else None
    inputs = (lambda x: _ssd_input(ssd, x)) if ssd is not None else (lambda x: x)
    example = (inputs(torch.rand(1, 3, img, img)),)
    with warnings.catch_warnings():                 # FX quantisation is deprecated upstream (torchao), still shipped
        warnings.filterwarnings("ignore", message=r".*(deprecated|quantized tensor creation functions)")
        prepared = prepare_fx(copy.deepcopy(DetectorTrunk(ssd) if ssd is not None else model).eval(),
                              _qconfig_mapping(engine, fast), example)
        with torch.no_grad():
            for b, (x, _) in enumerate(calib_loader):
                prepared(inputs(x))
                if b + 1 >= batches:
                    break
        quantized = convert_fx(prepared)
    if ssd is not None:
        quantized = QuantizedSSD(ssd, quantized, example[0])
    quantized.quant_engine = engine
    return quantized.eval()


def quantize_qat(model: torch.nn.Module, train_loader, backend: Optional[str] = None, epochs: int = 2, lr: float = 1e-4,
                 img: int = 128, max_batches: Optional[int] = None, log: Optional[Callable] = print) -> torch.nn.Module:
    """INT8 copy of `model` by quantisation-aware training (FX graph mode): a fake-quantised copy is fine-tuned
    on `train_loader` (AdamW; in the last epoch the batch-norm statistics and the activation ranges are frozen),
    then converted to the same INT8 kernels as `quantize_ptq`. For networks that post-training quantisation
    breaks, such as the ImageNet-initialised MobileNetV3s; torchvision trains its INT8 MobileNetV3 the same way."""
    from torch.ao.nn.intrinsic.qat import freeze_bn_stats
    from torch.ao.quantization import disable_observer, get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

    engine = set_quant_engine(backend)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*(deprecated|quantized tensor creation functions)")
        prepared = prepare_qat_fx(copy.deepcopy(model).train(), get_default_qat_qconfig_mapping(engine),
                                  (torch.rand(1, 3, img, img),))
        opt = torch.optim.AdamW(prepared.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            if epoch == epochs - 1 and epochs > 1:            # settle: fixed BN statistics and quantisation ranges
                prepared.apply(freeze_bn_stats)
                prepared.apply(disable_observer)
            prepared.train()
            total = 0.0
            for b, (x, y) in enumerate(train_loader):
                loss = loss_fn(prepared(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
                if max_batches and b + 1 >= max_batches:
                    break
            if log:
                log(f"    QAT epoch {epoch + 1}/{epochs}: train loss {total / (b + 1):.3f}")
        quantized = convert_fx(prepared.eval())
    quantized.quant_engine = engine
    return quantized.eval()


def load_quantized(model: torch.nn.Module, state_path, backend: Optional[str] = None, img: int = 128) -> torch.nn.Module:
    """INT8 model from a saved `quantize_ptq` or `quantize_qat` state: FX graph modules do not pickle, so the INT8 graph is
    rebuilt (uncalibrated) from the FP32 `model` and the calibrated weights and scales are loaded into it."""
    quantized = quantize_ptq(model, [(torch.rand(1, 3, img, img), None)], backend=backend, batches=1, img=img, fast=True)
    quantized.load_state_dict(torch.load(state_path, weights_only=False))
    return quantized
