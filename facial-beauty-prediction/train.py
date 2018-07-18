import torch
import torch.nn as nn
from torchvision.models import resnet50


class Beauty(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(True)
        self.predict = nn.Linear(1000, 1)

        self._reset_parameters()

    def forward(self, x):
        out = nn.functional.relu(self.resnet(x))
        score = self.predict(out)
        return score.squeeze()

    def _reset_parameters(self):
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.predict.weight.data.uniform_(-.1, .1)

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())


def train():
    model.train()
    total_loss = 0.

    for data, label in training_data:
        optimizer.zero_grad()

        scores = model(data)
        loss = criterion(scores, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data

    return total_loss[0] / training_data.ssz


def eval():
    model.eval()
    total_loss = 0.
    for data, label in validation_data:
        scores = model(data)
        loss = criterion(scores, label)
        total_loss += loss.data

    return total_loss[0] / validation_data.ssz


def main():
    best = None
    try:
        print('-' * 90)
        for epoch in range(1, args.epochs + 1):
            loss = train()
            print('| start of epoch {:3d}  | loss {:5.6f}'.format(epoch, loss))
            loss = eval()
            print('-' * 90)
            print('| end of epoch {:3d} | loss {:.4f}'.format(epoch, loss))
            print('-' * 90)
            if not best or best > loss:
                best = loss
                model_state_dict = model.state_dict()
                model_source = {
                    "settings": args,
                    "model": model_state_dict,
                }
                torch.save(model_source, args.save)
    except KeyboardInterrupt:
        print("-" * 90)
        print("Exiting from training early")


if __name__ == "__main__":
    import argparse
    from img_loader import Img_loader

    parser = argparse.ArgumentParser(description='Facial-Beau-Predict')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--unuse_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=514)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=str, default='./model.pt')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.unuse_cuda
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    training_data = Img_loader(
        "data/validation/train_1.txt", args.batch_size, is_cuda=use_cuda)
    validation_data = Img_loader(
        "data/validation/test_1.txt", args.batch_size, is_cuda=use_cuda, evaluation=True)

    model = Beauty()
    if use_cuda:
        model = model.cuda()

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=0.001)

    main()
