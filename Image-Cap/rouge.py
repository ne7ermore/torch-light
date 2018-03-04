import numpy as np
import torch
from torch.autograd import Variable

from const import PAD

def _lcs(x, y):
    n = len(x)
    m = len(y)
    table = dict()

    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
            else:
                table[i, j] = max(table[i-1, j], table[i, j-1])

    def recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i-1] == y[j-1]:
            return recon(i-1, j-1) + [x[i-1]]
        elif table[i-1, j] > table[i, j-1]:
            return recon(i-1, j)
        else:
            return recon(i, j-1)

    return len(recon(n, m)), n, m

def rouge_l(evals, refs):
    assert evals.size() == refs.size()
    use_cuda = evals.is_cuda

    evals, refs = map(lambda x: x.data.cpu().numpy(), [evals, refs])

    scores = []
    for eva, ref in zip(evals, refs):
        same_len, eva_len, ref_len = map(float,
                _lcs(eva[np.where(eva>PAD)], ref[np.where(ref>PAD)]))

        r_lcs, p_lcs = same_len/ref_len, same_len/eva_len

        beta = p_lcs / (r_lcs + 1e-12)
        f_lcs = ((1 + (beta**2)) * r_lcs * p_lcs) / (r_lcs + ((beta**2) * p_lcs) + 1e-12)
        scores.append(f_lcs)

    scores = np.asarray(scores, dtype=np.float32)
    scores = np.repeat(scores[:, np.newaxis], evals.shape[1], 1)
    scores = Variable(torch.from_numpy(scores))

    if use_cuda: scores = scores.cuda()

    return scores

def mask_score(props, words, scores):
    assert words.size() == scores.size()

    feats = props.size(2)
    mask = (words > 0).float()
    masked_ss = (scores*mask).view(-1, 1)
    masked_ss = masked_ss.repeat(1, feats)
    props = props.view(-1, feats)

    return props*masked_ss


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    data = Variable(torch.LongTensor([[3,1,2,3,1,0],[2,3,4,4,0,0]]))
    label = Variable(torch.LongTensor([[3,1,2,3,1,0],[2,3,2,3,1,0]]))
    data = data.cuda()
    label = label.cuda()
    print(rouge_l(data, label))

    props = torch.randn(16,17,256)
    words = torch.LongTensor([[i for i in range(16, -1, -1)] for _ in range(16)])
    scores = torch.randn(16,17)

    print(mask_score(props, words, scores))
