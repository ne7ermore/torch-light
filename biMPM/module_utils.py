import torch

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    """
    Return: [batch_size, decompse_dim, dim]
    """
    in_tensor = in_tensor.unsqueeze(1) #[batch_size, 'x', dim]
    decompose_params = decompose_params.unsqueeze(0) # [1, decompse_dim, dim]
    return torch.mul(in_tensor, decompose_params)

def max_repres(repre_cos):
    """
    Args:
        repre_cos - (q_repres, cos_simi_q)|(a_repres, cos_simi)
        Size: ([bsz, q_len, context_dim], [bsz, a_len, question_len])| ...
    Return:
        size - [bsz, a_len, context_dim] if question else [bsz, q_len, context_dim]
    """
    def tf_gather(input, index):
        """
        The same as tensorflow gather sometimes...
        Args:
            - input: dim - 3
            - index: dim - 2
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

    repres, cos_simi = repre_cos
    index = torch.max(cos_simi, 2)[1] # max_index
    return tf_gather(repres, index)

def cosine_cont(repr_context, relevancy, norm=False):
    """
    cosine siminlarity betwen context and relevancy
    Args:
        repr_context - [batch_size, other_len, context_lstm_dim]
        relevancy - [batch_size, this_len, other_len]
    Return:
        size - [batch_size, this_len, context_lstm_dim]
    """
    dim = repr_context.dim()

    temp_relevancy = relevancy.unsqueeze(dim) # [batch_size, this_len, other_len, 1]
    buff = repr_context.unsqueeze(1) # [batch_size, 1, other_len, context_lstm_dim]
    buff = torch.mul(buff, temp_relevancy) # [batch_size, this_len, other_len, context_lstm_dim]
    buff = buff.sum(2) # [batch_size, this_len, context_lstm_dim]
    if norm:
        relevancy = relevancy.sum(dim-1).clamp(min=1e-6) # [batch_size, this_len]
        relevancy = relevancy.unsqueeze(2) # [batch_size, this_len, 1]
        buff = buff.div(relevancy)
    return buff

if __name__ == "__main__":
    temp = np.zeros((1, 1))
    print(isinstance(temp, np.ndarray))
