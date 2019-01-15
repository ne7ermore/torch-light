import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationNet(nn.Module):
    def __init__(self, word_size, answer_size,
                 max_s_len, max_q_len, use_cuda,
                 story_len=20,
                 emb_dim=32,
                 story_hsz=32,
                 story_layers=1,
                 question_hsz=32,
                 question_layers=1):

        super().__init__()

        self.use_cuda = use_cuda

        self.max_s_len = max_s_len
        self.max_q_len = max_q_len

        self.story_len = story_len

        self.emb_dim = emb_dim
        self.story_hsz = story_hsz

        self.emb = nn.Embedding(word_size, emb_dim)
        self.story_rnn = torch.nn.LSTM(input_size=emb_dim,
                                       hidden_size=story_hsz,
                                       num_layers=story_layers,
                                       batch_first=True)
        self.question_rnn = torch.nn.LSTM(input_size=emb_dim,
                                          hidden_size=question_hsz,
                                          num_layers=question_layers,
                                          batch_first=True)

        self.g1 = nn.Linear((2*story_len)+(2*story_hsz)+question_hsz, 256)
        self.g2 = nn.Linear(256, 256)
        self.g3 = nn.Linear(256, 256)
        self.g4 = nn.Linear(256, 256)

        self.f1 = nn.Linear(256, 256)
        self.f2 = nn.Linear(256, 512)
        self.f3 = nn.Linear(512, answer_size)

        self._reset_parameters()

    def _reset_parameters(self, stddev=0.1):
        self.emb.weight.data.normal_(std=stddev)

        self.g1.weight.data.normal_(std=stddev)
        self.g1.bias.data.fill_(0)

        self.g2.weight.data.normal_(std=stddev)
        self.g2.bias.data.fill_(0)

        self.g3.weight.data.normal_(std=stddev)
        self.g3.bias.data.fill_(0)

        self.g4.weight.data.normal_(std=stddev)
        self.g4.bias.data.fill_(0)

        self.f1.weight.data.normal_(std=stddev)
        self.f1.bias.data.fill_(0)

        self.f2.weight.data.normal_(std=stddev)
        self.f2.bias.data.fill_(0)

        self.f3.weight.data.normal_(std=stddev)
        self.f3.bias.data.fill_(0)

    def g_theta(self, x):
        x = F.relu_(self.g1(x))
        x = F.relu_(self.g2(x))
        x = F.relu_(self.g3(x))
        x = F.relu_(self.g4(x))
        return x

    def init_tags(self):
        tags = torch.zeros((self.story_len, self.story_len))
        if self.use_cuda:
            tags = tags.cuda()
        for i in range(self.story_len):
            tags[i, i].fill_(1)
        return tags

    def forward(self, story, question):
        tags = self.init_tags()
        bsz = story.shape[0]

        s_emb = self.emb(story)
        s_emb = s_emb.view(-1, self.max_s_len, self.emb_dim)

        _, (s_state, _) = self.story_rnn(s_emb)
        s_state = s_state[-1, :, :]
        s_state = s_state.view(-1, self.story_len, self.story_hsz)

        s_tags = tags.unsqueeze(0)
        s_tags = s_tags.repeat((bsz, 1, 1))

        story_objects = torch.cat((s_state, s_tags), dim=2)

        q_emb = self.emb(question)
        _, (q_state, _) = self.question_rnn(q_emb)
        q_state = q_state[-1, :, :]

        sum_g_theta = 0
        for i in range(self.story_len):
            this_tensor = story_objects[:, i, :]
            for j in range(self.story_len):
                u = torch.cat(
                    (this_tensor, story_objects[:, j, :], q_state), dim=1)
                g = self.g_theta(u)
                sum_g_theta = torch.add(sum_g_theta, g)

        out = F.relu(self.f1(sum_g_theta))
        out = F.relu(self.f2(out))
        out = self.f3(out)

        return out


if __name__ == "__main__":
    from data_loader import DataLoader

    data = torch.load("data/corpus.pt")

    training_data = DataLoader(data["story"],
                               data["question"],
                               data["answer"],
                               data["max_q_len"],
                               data["max_s_len"],
                               data["word2idx"],
                               data["answer2idx"],
                               cuda=True)

    s, q, a = next(training_data)

    model = RelationNet(len(data["word2idx"]),
                        len(data["answer2idx"]),
                        data["max_s_len"],
                        data["max_q_len"],
                        True,
                        story_hsz=32,
                        story_layers=1,
                        question_hsz=32,
                        question_layers=1)
    model.cuda()
    t = model(s, q)
