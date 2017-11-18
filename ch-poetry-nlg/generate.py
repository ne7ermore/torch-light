import torch
from torch.autograd import Variable

from model import Model

class Generate:
    def __init__(self, model=None, model_source=None, src_dict=None, args=None):
        assert model is not None or model_source is not None

        if model is None:
            model_source = torch.load(model_source, map_location=lambda storage, loc: storage)
            self.dict = model_source["src_dict"]
            self.args = model_source["settings"]
            model = Model(self.args)
            model.load_state_dict(model_source['model'])
        else:
            self.dict = src_dict
            self.args = args

        self.num_directions = 2 if self.args.bidirectional else 1
        self.idx2word = {v: k for k, v in self.dict.items()}
        self.model = model.eval()

    def Create(self, max_len):
        args = self.args

        num_layers = args.lstm_layers*self.num_directions
        hidden = self.model.init_hidden(1)

        # random sample
        prob = torch.rand(1).mul(args.vocab_size).long()
        input = Variable(prob.unsqueeze(1), volatile=True)
        portry = self.idx2word[prob.tolist()[0]]

        count = 1
        for _ in range(1, max_len):
            output, hidden = self.model(input, hidden)
            prob = output.squeeze().data
            next_word = torch.max(prob, -1)[1].tolist()[0]
            input.data.fill_(next_word)
            if count == 4:
                portry += self.idx2word[next_word]
                portry += "，"
                count = 0
            else:
                portry += self.idx2word[next_word]
                count += 1
        portry = portry[:-1] + "。"
        return portry

if __name__ == "__main__":
    G = Generate(model_source="ch_poe.pt")
    print(G.Create(20))
