import torch
import torch.nn as nn

import copy

class GramMatrix(nn.Module):
    def forward(self, input):
        _, channels, h, w = input.size()
        out = input.view(-1, h*w)
        out = torch.mm(out, out.t())
        return out.div(channels*h*w)

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super().__init__()

        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()
        self.gm = GramMatrix()

    def forward(self, input):
        gm = self.gm(input.clone())
        loss = self.criterion(gm*self.weight, self.target)
        return loss

def check_layers(layers):
    """
    relu1_* - 2， relu2_* - 2， relu3_* - 4， relu4_* - 4， relu5_* - 4
    """
    in_layers = []
    for layer in layers:
        layer = layer[-3:]
        if layer[0] == '1' or layer[0] == '2':
            in_layers += [2*(int(layer[0])-1) + int(layer[2])]
        else:
            in_layers += [4*(int(layer[0])-3) + int(layer[2]) + 4]
    return in_layers

class Vgg_Model(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.layers = copy.deepcopy(vgg)

    def forward(self, input, out_layers):
        relu_outs, out = [], input
        out_layers = check_layers(out_layers)

        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                relu_outs.append(out)

        outs = [relu_outs[index-1] for index in out_layers]
        return outs

if __name__ == '__main__':
    from torch.autograd import Variable

    from torchvision.models import vgg19
    from img_loader import IMG_Processer

    STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    CONTENT_LAYERS = ('relu4_2',)
    vgg = vgg19(True).features
    vm = Vgg_Model(vgg)
    vm = vm.cuda()

    ip = IMG_Processer()
    _style, _content = ip.img2tensor('vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg')
    _style = Variable(_content.unsqueeze(0))
    _style = _style.cuda()
    # print(vm(_style, STYLE_LAYERS))
    print(vm(_style, STYLE_LAYERS+CONTENT_LAYERS))
