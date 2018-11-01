import torch
import torch.nn as nn
import torch.nn.functional as F


class C_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        # [4*out_feature, in_feature]
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        # [4*out_feature, out_feature]
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.b_ih = None
            self.b_hh = None

        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = F.linear(input, self.w_ih, self.b_ih) + \
            F.linear(hx, self.w_hh, self.b_hh)  # [bsz, 4*hidden_size]
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        in_gate, forget_gate, out_gate = map(
            F.sigmoid, [in_gate, forget_gate, out_gate])
        cell_gate = F.tanh(cell_gate)

        cy = forget_gate * cx + in_gate * cell_gate
        hy = out_gate * F.tanh(cy)

        return hy, cy
