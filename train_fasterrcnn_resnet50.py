from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class YoloDetDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str):
        self.dataset_root = dataset_root
        self.img_dir = dataset_root / "images" / split
        self.lbl_dir = dataset_root / "labels" / split
        self.images = sorted(
            [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in self.img_dir.glob(ext)],
            key=lambda p: p.name,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_path = self.lbl_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 5:
                        continue
                    _, x, y, bw, bh = p[:5]
                    x, y, bw, bh = float(x), float(y), float(bw), float(bh)
                    x1 = (x - bw / 2) * w
                    y1 = (y - bh / 2) * h
                    x2 = (x + bw / 2) * w
                    y2 = (y + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if len(boxes_t) else torch.zeros((0,))
        iscrowd = torch.zeros((len(labels_t),), dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([h, w]),
        }
        image_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return image_t, target, img_path


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return images, targets, paths


def build_model(num_classes: int = 2):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    total = 0.0
    for images, targets, _ in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def evaluate_coco(model, loader, device, out_dir: Path) -> dict:
    model.eval()
    coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "apple"}]}
    coco_dt = []
    ann_id = 1
    infer_times = []

    with torch.no_grad():
        for images, targets, paths in loader:
            images_dev = [img.to(device) for img in images]
            t0 = time.time()
            outputs = model(images_dev)
            infer_times.append((time.time() - t0) * 1000 / max(1, len(images_dev)))

            for out, tgt, path in zip(outputs, targets, paths):
                h, w = map(int, tgt["orig_size"].tolist())
                image_id = int(tgt["image_id"].item())
                coco_gt["images"].append({"id": image_id, "file_name": path.name, "width": w, "height": h})

                for b in tgt["boxes"].numpy():
                    x1, y1, x2, y2 = map(float, b)
                    coco_gt["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()
                for b, s, lb in zip(boxes, scores, labels):
                    if int(lb) != 1:
                        continue
                    x1, y1, x2, y2 = map(float, b)
                    coco_dt.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(s),
                        }
                    )

    gt_json = out_dir / "fasterrcnn_gt.json"
    dt_json = out_dir / "fasterrcnn_dt.json"
    gt_json.write_text(json.dumps(coco_gt), encoding="utf-8")
    dt_json.write_text(json.dumps(coco_dt), encoding="utf-8")

    coco = COCO(str(gt_json))
    dt = coco.loadRes(str(dt_json))
    evaluator = COCOeval(coco, dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    return {
        "precision": float(stats[1]),  # AP50 proxy
        "recall": float(stats[8]),  # AR@100
        "map50": float(stats[1]),
        "map50_95": float(stats[0]),
        "inference_ms": float(np.mean(infer_times) if infer_times else 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Standard Faster R-CNN ResNet50-FPN baseline.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg["path"]).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = YoloDetDataset(dataset_root, "train")
    val_set = YoloDetDataset(dataset_root, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate_fn)

    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device.split(',')[0]}")

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs} loss={loss:.4f}")

    metrics = evaluate_coco(model, val_loader, device, out_dir)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "fasterrcnn_resnet50_final.pth")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
