import argparse
import os

import torch
import torch.optim as optim
import numpy as np

from darknet import DarkNet
from img_loader import Data_loader

from utils import load_classes, predict, evaluate

parser = argparse.ArgumentParser(description='YOLO-v3 Train')
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument('--img_size', type=int, default=416)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--max_objects', type=int, default=50)
parser.add_argument("--confidence", type=float, default=0.5)
parser.add_argument("--nms_conf", type=float, default=0.45)


def train(folder="weights"):
    os.makedirs(folder, exist_ok=True)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    classes = load_classes()
    num_classes = len(classes)

    model = DarkNet(use_cuda, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    training_data = Data_loader(
        "data/labels/train2014/",
        "data/train2014",
        img_size=args.img_size,
        max_objects=args.max_objects,
        batch_size=args.batch_size,
        is_cuda=use_cuda)

    validation_data = Data_loader(
        "data/labels/val2014/",
        "data/val2014",
        img_size=args.img_size,
        max_objects=args.max_objects,
        batch_size=args.batch_size,
        is_cuda=use_cuda)

    for epoch in range(args.epoch):

        model.train()
        for batch_i, (imgs, labels) in enumerate(training_data):
            optimizer.zero_grad()
            loss, gather_losses = model(imgs, labels)
            loss.backward()
            optimizer.step()

            print(f"""[Epoch {epoch+1}/{args.epoch},Batch {batch_i+1}/{training_data.stop_step}] [Losses: x {gather_losses["x"]:.5f}, y {gather_losses["y"]:.5f}, w {gather_losses["w"]:.5f}, h { gather_losses["h"]:.5f}, conf {gather_losses["conf"]:.5f}, cls {gather_losses["cls"]:.5f}, total {loss.item():.5f}, recall: {gather_losses["recall"]:.5f}, precision: {gather_losses["precision"]:.5f}]""")

        torch.save({"model": model.state_dict(),
                    "classes": classes}, f"{folder}/{epoch}.weights.pt")

        all_detections = []
        all_annotations = []

        model.eval()
        for imgs, labels in validation_data:
            with torch.no_grad():
                prediction = model(imgs)
                outputs = predict(prediction, args.nms_conf, args.confidence)

            labels = labels.cpu()
            for output, annotations in zip(outputs, labels):
                all_detections.append([np.array([])
                                       for _ in range(num_classes)])
                if output is not None:
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()
                    pred_labels = output[:, -1].cpu().numpy()

                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    for label in range(num_classes):
                        all_detections[-1][label] = pred_boxes[pred_labels == label]

                all_annotations.append([np.array([])
                                        for _ in range(num_classes)])

                if any(annotations[:, -1] > 0):
                    annotation_labels = annotations[annotations[:, -1]
                                                    > 0, 0].numpy()
                    _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = _annotation_boxes[:,
                                                               0] - _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 1] = _annotation_boxes[:,
                                                               1] - _annotation_boxes[:, 3] / 2
                    annotation_boxes[:, 2] = _annotation_boxes[:,
                                                               0] + _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 3] = _annotation_boxes[:,
                                                               1] + _annotation_boxes[:, 3] / 2
                    annotation_boxes *= args.img_size

                    for label in range(num_classes):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

        average_precisions = evaluate(
            num_classes, all_detections, all_annotations)

        print(f"""{"-"*40}evaluation.{epoch}{"-"*40}""")
        for c, ap in average_precisions.items():
            print(f"Class '{c}' - AP: {ap}")

        mAP = np.mean(list(average_precisions.values()))
        print(f"mAP: {mAP}")
        print(f"""{"-"*40}end{"-"*40}""")


if __name__ == "__main__":
    train()
