import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

EPOCHS = 100

test_sentence = """n-gram models are widely used in statistical natural
language processing . In speech recognition , phonemes and sequences of
phonemes are modeled using a n-gram distribution . For parsing , words
are modeled such that each n-gram is composed of n words . For language
identification , sequences of characters / graphemes ( letters of the alphabet
) are modeled for different languages For sequences of characters ,
the 3-grams ( sometimes referred to as " trigrams " ) that can be
generated from " good morning " are " goo " , " ood " , " od " , " dm ",
" mo " , " mor " and so forth , counting the space character as a gram
( sometimes the beginning and end of a text are modeled explicitly , adding
" __g " , " _go " , " ng_ " , and " g__ " ) . For sequences of words ,
the trigrams that can be generated from " the dog smelled like a skunk "
are " # the dog " , " the dog smelled " , " dog smelled like ", " smelled
like a " , " like a skunk " and " a skunk # " .""".split()

trigrams = [([test_sentence[i], test_sentence[i+1]],
            test_sentence[i+2]) for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)

word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, context_size=2):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.l1 = nn.Linear(context_size * embedding_dim, 128)
        self.l2 = nn.Linear(128, vocab_size)
        self._init_weight()

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.l1(embeds))
        out = self.l2(out)
        log_probs = F.log_softmax(out)
        return log_probs

    def _init_weight(self, scope=0.1):
        self.embeddings.weight.data.uniform_(-scope, scope)
        self.l1.weight.data.uniform_(0, scope)
        self.l1.bias.data.fill_(0)
        self.l2.weight.data.uniform_(0, scope)
        self.l2.bias.data.fill_(0)

criterion = nn.NLLLoss()
model = NGram(len(vocab))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        context_idxs = list(map(lambda w: word2idx[w], context))
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        model.zero_grad()

        log_probs = model(context_var)
        loss = criterion(log_probs,
            autograd.Variable(torch.LongTensor([word2idx[target]])))

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    print(total_loss[0])

model.eval()
def predict(context):
    context_idxs = list(map(lambda w: word2idx[w], context))
    context_var = autograd.Variable(
        torch.LongTensor(context_idxs), volatile=True)

    predict = model(context_var)
    index = (torch.max(predict, 1)[1]).data.tolist()[0]
    return idx2word[index]


for context in [["widely", "used"], ["and", "so"], ["are", "modeled"]]:
    print("{} + {} = {}".format(context[0], context[1], predict(context)))

