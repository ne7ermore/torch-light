import argparse

parser = argparse.ArgumentParser(description='Neural Artistic Style')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs for train')
parser.add_argument('--style_layers', type=str, default="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1")
parser.add_argument('--content_layers', type=str, default="relu4_2")
parser.add_argument('--model', type=str, default="vgg19")
parser.add_argument('--style_img', type=str, default="night.jpg")
parser.add_argument('--content_img', type=str, default="WechatIMG742.jpeg")
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--beta', type=float, default=1e3)

args = parser.parse_args()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg19

from img_loader import IMG_Processer
from model import Vgg_Model, StyleLoss, GramMatrix

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and not args.unuse_cuda
args.style_layers = list(args.style_layers.split(','))
args.content_layers = list(args.content_layers.split(','))
out_layers = args.style_layers + args.content_layers

if use_cuda:
    torch.cuda.manual_seed(args.seed)

vgg = vgg19(True).features
model = Vgg_Model(vgg)

ip = IMG_Processer()
style_input, content_input = ip.img2tensor(args.style_img, args.content_img)
style_input, content_input = Variable(style_input.unsqueeze(0)), Variable(content_input.unsqueeze(0))

if use_cuda:
    style_input, content_input = map(lambda v: v.cuda(), (style_input, content_input))
    model = model.cuda()

out_img = content_input.clone()
out_param = nn.Parameter(out_img.data)

_content_layers = [layer.detach() for layer in model(content_input, args.content_layers)]
_style_layers = model(style_input, args.style_layers)
style_criterions = [StyleLoss(GramMatrix()(layer), args.beta) for layer in _style_layers]

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS([out_param]);
n_epoch = [0]

def closure():

    optimizer.zero_grad()

    _out_layers = model(out_param, out_layers)

    _loss = 0
    for step, _criterion in enumerate(style_criterions):
        _loss += _criterion(_out_layers[step])

    for step, _out_layer in enumerate(_out_layers[step+1:]):
        _loss += criterion(_out_layer*args.alpha, _content_layers[step]*args.alpha)

    _loss.backward()
    n_epoch[0] += 1

    if n_epoch[0] % 30 == 0:
        print('epochs: {}, loss: {}'.format(n_epoch[0], _loss.data[0]))
        ip.tensor2img(out_param.data.cpu().squeeze(), n_epoch[0])

    return _loss

while n_epoch[0] < args.epochs:
    optimizer.step(closure)
