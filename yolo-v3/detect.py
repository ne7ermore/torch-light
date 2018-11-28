import argparse

import torch

from darknet import DarkNet
from img_loader import IMGProcess

parser = argparse.ArgumentParser(description='YOLO-v3 Detect')
parser.add_argument("--images", type=str, default='imgs')
parser.add_argument("--result", type=str, default="result")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--img_size', type=int, default=416)
parser.add_argument('--confidence', type=float, default=0.5)
parser.add_argument('--nms_thresh', type=float, default=0.4)
parser.add_argument("--weights", type=str, default="yolo.v3.coco.weights.pt")
parser.add_argument('--no_cuda', action='store_true')


def main():
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    model_source = torch.load(args.weights)
    model = DarkNet(use_cuda,model_source["num_classes"])
    model.load_state_dict(model_source['model'])
    model.eval()
    if use_cuda:
        model = model.cuda()

    ip = IMGProcess(model_source,
                    use_cuda=use_cuda,
                    img_path=args.images,
                    img_size=args.img_size,
                    confidence=args.confidence,
                    result=args.result)

    print("-" * 57 + "Result" + "-" * 57)
    for batch in ip:
        outputs = ip.predict(model(batch), nms_conf=args.nms_thresh)
        for name, objs in outputs:
            print("Image - {}".format(name))
            print("Detect Objects - [{}]".format(", ".join(objs)))
            print("-" * 120)


if __name__ == "__main__":
    main()
