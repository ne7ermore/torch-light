import torch

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    """
    Return: [bsz, decompse_dim, dim]
    """
    in_tensor = in_tensor.unsqueeze(1) #[bsz, 1, dim]
    decompose_params = decompose_params.unsqueeze(0) # [1, decompse_dim, dim]
    return torch.mul(in_tensor, decompose_params)

def tf_gather(input, index):
    """
    The same as tensorflow gather sometimes...
    Return: [input.size(0), index.size(1), input.size(2)]
    """
    bsz = input.size(0)
    sent_size = input.size(1)
    dim_size = input.size(2)
    for n, i in enumerate(index):
        index.data[n] = i.data.add(n*sent_size)

    input = input.view(-1, dim_size)
    index = index.view(-1)

    temp = input.index_select(0 ,index)
    return temp.view(bsz, -1, dim_size)

def cosine_cont(repr_context, relevancy, norm=False):
    """
    cosine siminlarity betwen context and relevancy
    Args:
        repr_context - [bsz, other_len, context_lstm_dim]
        relevancy - [bsz, this_len, other_len]
    Return:
        size - [bsz, this_len, context_lstm_dim]
    """
    dim = repr_context.dim()

    temp_relevancy = relevancy.unsqueeze(dim) # [bsz, this_len, other_len, 1]
    buff = repr_context.unsqueeze(1) # [bsz, 1, other_len, context_lstm_dim]
    buff = torch.mul(buff, temp_relevancy) # [bsz, this_len, other_len, context_lstm_dim]
    buff = buff.sum(2) # [bsz, this_len, context_lstm_dim]
    if norm:
        relevancy = relevancy.sum(dim-1).clamp(min=1e-6) # [bsz, this_len]
        relevancy = relevancy.unsqueeze(2) # [bsz, this_len, 1]
        buff = buff.div(relevancy)
    return buff
