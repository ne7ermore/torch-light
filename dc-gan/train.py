import argparse

parser = argparse.ArgumentParser(description='DC-Gan')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--unuse_cuda', action='store_true')
parser.add_argument('--path', type=str, default='data/')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--relu_leak', type=float, default=0.2)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--channel_dims', type=str, default='512,256,128,64')

args = parser.parse_args()

import torch

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda

args.channel_dims = list(map(int, args.channel_dims.split(',')))

if use_cuda:
    torch.cuda.manual_seed(args.seed)

# ##############################################################################
# Load data
################################################################################
from data_loader import Data_loader

real_datas = Data_loader('data/', args.img_size, args.batch_size, use_cuda)

# ##############################################################################
# Build model
# ##############################################################################
import model

G = model.Generator(args.img_size, args.img_size, args.channel_dims, args.z_dim)
D = model.Discriminator(args.img_size, args.img_size, args.channel_dims, args.relu_leak)
if use_cuda:
    G, D = G.cuda(), D.cuda()

optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
criterion = torch.nn.BCELoss()

# ##############################################################################
# Training
# ##############################################################################
real_label, fake_label = 1, 0

def train():
    loss_g = loss_d = 0
    for real_data in tqdm(real_datas, mininterval=1,
            desc='Train Processing', leave=False):
        D.zero_grad()
        real_out = D(real_data)
        real_target = Variable(real_data.data.new(args.batch_size, 1).fill_(real_label))
        real_loss_d = criterion(real_out, real_target)
        real_loss_d.backward()

        z_input = model.gen_z(args.batch_size, args.z_dim, use_cuda)
        g_out = G(z_input)
        fake_out = D(g_out.detach())
        fake_target = Variable(real_data.data.new(args.batch_size, 1).fill_(fake_label))
        fake_loss_d = criterion(fake_out, fake_target)
        fake_loss_d.backward()
        optimizer_D.step()

        G.zero_grad()
        out = D(g_out)
        target = Variable(real_data.data.new(args.batch_size, 1).fill_(real_label))
        loss = criterion(out, target)
        loss.backward()
        optimizer_G.step()

        loss_d += fake_loss_d.data + real_loss_d.data
        loss_g += loss.data

    return loss_g[0], loss_d[0]

from torch.autograd import Variable        
from tqdm import tqdm
import time

loss_ds, loss_gs = [], []
for epoch in range(1, args.epochs+1):
    start_time = time.time()
    G.train()
    D.train()
    loss_g, loss_d = train()

    print('| epoch {:3d} | time: {:2.2f}s | D loss {:5.6f} | G loss {:5.6f}'.format(epoch, time.time() - start_time, loss_d, loss_g))
    loss_ds.append(loss_d)
    loss_gs.append(loss_g)
    if epoch % 2 == 0:
        G.eval()
        g_out = G(model.gen_z(64, args.z_dim, use_cuda, True))
        real_datas.gen_image(g_out, epoch)