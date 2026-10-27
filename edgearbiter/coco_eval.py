"""COCO box mAP of a torchvision detector (FP32, or the post-training INT8 `quantize.QuantizedSSD`) on val2017.

    data/coco/annotations/instances_val2017.json, data/coco/val2017/*.jpg   (images.cocodataset.org)

torchvision's detection models predict COCO category ids, so their boxes go to pycocotools unchanged. The
models keep their own thresholds (SSDLite: score > 0.001, at most 300 boxes per image), as in torchvision's
reference evaluation.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch

VAL_IMAGES = "https://images.cocodataset.org/zips/val2017.zip"                           # 5,000 images, ~0.8 GB
ANNOTATIONS = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"  # ~0.25 GB


def coco_val(root: str = "data/coco", download: bool = True, log: Optional[Callable] = print) -> Tuple[Path, Path]:
    """Annotation file and image folder of COCO val2017 under `root`, downloaded and extracted if missing."""
    import urllib.request
    root_path = Path(root)
    ann, images = root_path / "annotations" / "instances_val2017.json", root_path / "val2017"
    for target, url, members in ((ann, ANNOTATIONS, ("annotations/instances_val2017.json",)), (images, VAL_IMAGES, None)):
        if target.exists():
            continue
        if not download:
            raise FileNotFoundError(f"{target} not found (download it from {url})")
        root_path.mkdir(parents=True, exist_ok=True)
        archive = root_path / Path(url).name
        if not archive.exists():
            if log:
                log(f"[coco] downloading {url}")
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root_path, members=members)
        archive.unlink()
    return ann, images


@torch.no_grad()
def coco_map(model: torch.nn.Module, ann_file, image_dir, max_images: Optional[int] = None, batch: int = 8,
             log: Optional[Callable] = print) -> dict:
    """mAP@[.5:.95], mAP@.5 and mAP@.75 of `model` (torchvision detection interface) on the annotated images."""
    import contextlib
    import io

    import torchvision.transforms.functional as F
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_file))
    ids = sorted(gt.getImgIds())[:max_images]
    model.eval()
    detections, t0 = [], time.time()
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        images = [F.to_tensor(Image.open(Path(image_dir) / gt.loadImgs(i)[0]["file_name"]).convert("RGB")) for i in chunk]
        for image_id, out in zip(chunk, model(images)):
            for (x1, y1, x2, y2), score, label in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
                detections.append({"image_id": image_id, "category_id": int(label), "bbox": [x1, y1, x2 - x1, y2 - y1],
                                   "score": float(score)})
        if log and (start // batch) % 100 == 0:
            log(f"[coco] {start + len(chunk)} / {len(ids)} images, {time.time() - t0:.0f} s")
    seconds = time.time() - t0
    stats = [0.0] * 12
    if detections:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = COCOeval(gt, gt.loadRes(detections), "bbox")
            evaluator.params.imgIds = ids
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        stats = [float(v) for v in evaluator.stats]
    return {"map": stats[0], "map50": stats[1], "map75": stats[2], "map_small": stats[3], "map_medium": stats[4],
            "map_large": stats[5], "images": len(ids), "detections": len(detections), "seconds": round(seconds, 1)}
