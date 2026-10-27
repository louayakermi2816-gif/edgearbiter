"""COCO mAP plumbing on a two-image synthetic dataset (no download): a detector that returns the ground truth
scores 1.0, one that misses a box scores less."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from edgearbiter.coco_eval import coco_map

BOXES = {1: [(1, [10, 20, 30, 40])], 2: [(1, [5, 5, 20, 20]), (3, [40, 30, 50, 25])]}   # image -> (category, xywh)


def dataset(tmp_path):
    from PIL import Image
    images, annotations = [], []
    for image_id, boxes in BOXES.items():
        Image.new("RGB", (100, 80), (image_id * 60, 90, 120)).save(tmp_path / f"{image_id}.jpg")
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": 100, "height": 80})
        for category, box in boxes:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category, "bbox": box,
                                "area": box[2] * box[3], "iscrowd": 0})
    ann = tmp_path / "instances.json"
    ann.write_text(json.dumps({"images": images, "annotations": annotations,
                               "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]}))
    return ann


class Oracle(torch.nn.Module):
    """Returns the annotated boxes of each image (identified by its colour), optionally dropping the cars."""

    def __init__(self, drop_cars=False):
        super().__init__()
        self.drop_cars = drop_cars

    def forward(self, images):
        out = []
        for img in images:
            image_id = round(float(img[0, 0, 0]) * 255 / 60)
            boxes = [(c, b) for c, b in BOXES[image_id] if not (self.drop_cars and c == 3)]
            out.append({"boxes": torch.tensor([[x, y, x + w, y + h] for _, (x, y, w, h) in boxes], dtype=torch.float32),
                        "scores": torch.full((len(boxes),), 0.9), "labels": torch.tensor([c for c, _ in boxes])})
        return out


def test_perfect_detections_score_one_and_a_missed_class_scores_less(tmp_path):
    ann = dataset(tmp_path)
    perfect = coco_map(Oracle(), ann, tmp_path, log=None)
    assert perfect["map"] == pytest.approx(1.0) and perfect["images"] == 2 and perfect["detections"] == 3
    missing = coco_map(Oracle(drop_cars=True), ann, tmp_path, max_images=2, batch=1, log=None)
    assert missing["map"] == pytest.approx(0.5)                     # mean over two categories, one never detected
