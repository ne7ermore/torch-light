import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNN_Text_C(nn.Module):

    def __init__(self, args):
        super(CNN_Text_C, self).__init__()
        assert len(args.hidden_size) == len(args.dropout_switches)

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.lookup_table = nn.Embedding(self.tokens, self.embed_dim)
        self.init_embedding()

        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                            nn.Conv2d(in_channels=1,
                            out_channels=self.kernel_num,
                            kernel_size=(filter_size, self.embed_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))

        self.hid_layers = []
        ins = len(self.filter_sizes) * self.kernel_num
        for i, hid_size in enumerate(self.hidden_size):
            hid_attr_name = "hid_layer_%d" % i
            self.__setattr__(hid_attr_name, nn.Linear(ins, hid_size))
            self.hid_layers.append(self.__getattr__(hid_attr_name))
            ins = hid_size

        self.logistic = nn.Linear(ins, self.class_num)

    def forward(self, x):
        """
        :param x:
                input x is in size of [N, C, H, W]
                N: batch size
                C: number of channel, in text case, this is 1
                H: height, in text case, this is the length of the text
                W: width, in text case, this is the dimension of the embedding
        :return:
                a tensor [N, L], where L is the number of classes
        """
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        # lookup table output size [N, H, W=emb_dim]
        x = self.lookup_table(x)
        # expand x to [N, 1, H, W=emb_dim]
        x = x.unsqueeze(c_idx)
        enc_outs = []
        for encoder in self.encoders:
            enc_ = F.relu(encoder(x))
            k_h = enc_.size()[h_idx]
            k_w = 1
            enc_ = F.max_pool2d(enc_, kernel_size=(k_h, k_w))
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)
        # each of enc_outs size [N, C]
        encoding = torch.cat(enc_outs, 1)
        hid_in = encoding
        for hid_layer, do_dropout in zip(self.hid_layers, self.dropout_switches):
            hid_out = F.relu(hid_layer(hid_in))
            if do_dropout == "t":
                hid_out = F.dropout(hid_out)
            hid_in = hid_out
        # pred_prob = F.log_softmax(self.logistic(hid_in))
        pred_prob = self.logistic(hid_in) # CrossEntropyLoss = logsoftmax + nullloss
        return pred_prob

    def init_embedding(self):
        self.lookup_table.weight.data.uniform_(-0.1, 0.1)

