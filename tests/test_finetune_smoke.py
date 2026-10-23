"""The fine-tuning loop runs end to end on synthetic images (no download) and writes its artefacts."""
import json

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("torchvision.datasets")

from edgearbiter.data import eval_transform
from edgearbiter.finetune import FinetuneConfig, build_model, checkpoint_path, read_metrics, train


def test_train_from_scratch_on_fake_data(tmp_path):
    ds = datasets.FakeData(size=16, image_size=(3, 80, 80), num_classes=10, transform=eval_transform(64))
    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    cfg = FinetuneConfig(epochs=2, img=64, pretrained=False, out_dir=str(tmp_path))
    metrics = train(build_model(pretrained=False), loader, loader, cfg, log=None)
    ckpt = checkpoint_path(str(tmp_path), "scratch")
    assert ckpt.name == "mobilenet_v3_small_imagenette_scratch.pt" and ckpt.exists()
    assert read_metrics(ckpt)["init"] == "scratch" and len(metrics["history"]) == 2
    assert 0.0 <= metrics["best_val_top1"] <= 1.0
    model = build_model(pretrained=False); model.load_state_dict(torch.load(ckpt))
    assert json.loads(ckpt.with_suffix(".json").read_text())["config"]["img"] == 64
