import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from base_layer import *
from module_utils import *

# place filled defined
PF_POS = 1

class biMPModule(nn.Module):
    """
    Word Representation Layer
        - Corpus Embedding(Word Embedding)
        - Word Embedding(Character Embedding)
    from biMPM TensorFlow, there is a layer called Highway which is a f**king lstmcell implement, do not know y?
    Context Representation Layer
    Matching Layer
    Aggregation Layer
    Prediction Layer
    """
    def __init__(self, args):
        super().__init__()

        # init argments
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        # context repres -> matching | W
        self.context_dim = self.corpus_emb_dim + self.word_lstm_dim

        # Word Representation Layer - Corpus Embedding(Word Embedding)
        self.corpus_emb = nn.Embedding(self.corpora_len, self.corpus_emb_dim)
        self._init_corpus_embedding()

        # Word Representation Layer - Word Embedding(Character Embedding)
        self.word_emb = nn.Embedding(self.words_len, self.word_emb_dim)
        # self.word_lstm_cell = nn.LSTMCell(self.word_emb_dim, self.word_lstm_dim)
        self.word_lstm = nn.LSTM(self.word_emb_dim,
                                 self.word_lstm_dim,
                                 num_layers=self.word_layer_num,
                                 dropout=self.dropout,
                                 batch_first=True)

        # Context Representation Layer
        self.context_lstm = nn.LSTM(self.context_dim,
                                    self.context_lstm_dim,
                                    num_layers=self.context_layer_num,
                                    dropout=self.dropout,
                                    batch_first=True,
                                    bidirectional=True)

        # Pre Attentive Matching
        self.pre_q_attmath_layer = AtteMatchLay(self.mp_dim, self.context_dim)
        self.pre_a_attmath_layer = AtteMatchLay(self.mp_dim, self.context_dim)

        # Full Matching
        self.f_full_layer = FullMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_full_layer = FullMatchLay(self.mp_dim, self.context_lstm_dim)

        # Maxpoll Matching
        self.f_max_layer = MaxpoolMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_max_layer = MaxpoolMatchLay(self.mp_dim, self.context_lstm_dim)

        # Attentive Matching
        self.f_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)

        # Max Attentive Matching
        self.f_max_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_max_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)

        # Aggregation Layer
        self.aggre_lstm = nn.LSTM(11*self.mp_dim+6,
                                  self.aggregation_lstm_dim,
                                  num_layers=self.aggregation_layer_num,
                                  dropout=self.dropout,
                                  batch_first=True,
                                  bidirectional=True)
        # Prediction Layer
        self.l1 = nn.Linear(self.aggregation_lstm_dim*4,
                            self.aggregation_lstm_dim*2)

        self.l2 = nn.Linear(self.aggregation_lstm_dim*2,
                            self.num_class)

    def forward(self, q_corpora, q_words, a_corpora, a_words):
        """
        Module main forward
        Args:
            - q_corpora
            - q_words
            - a_corpora
            - a_words
        """
        # Step 1 - Get Mask from q_corpora and a_corpora
        self.q_mask = q_corpora.ge(PF_POS)
        self.a_mask = a_corpora.ge(PF_POS)

        # Step 2 - Word Representation Layer
        self.q_repres = self._word_repre_layer((q_corpora, q_words))
        self.a_repres = self._word_repre_layer((a_corpora, a_words))

        # Step 3 - Cosine Similarity and mask
        iqr_temp = self.q_repres.unsqueeze(1) # [bsz, 1, q_len, context_dim]
        ipr_temp = self.a_repres.unsqueeze(2) # [bsz, a_len, 1, context_dim]

        # [bsz, a_len, q_len]
        simi = F.cosine_similarity(iqr_temp, ipr_temp, dim=(iqr_temp.dim()-1))
        simi_mask = self._cosine_similarity_mask(simi)

        # Step 4 - Matching Layer
        q_aware_reps, a_aware_reps = self._bilateral_match(simi_mask)
        q_aware_reps = F.dropout(q_aware_reps, p=self.dropout)
        a_aware_reps = F.dropout(a_aware_reps, p=self.dropout)

        # Step 5 - Aggregation Layer
        aggre = self._aggre(q_aware_reps, a_aware_reps)

        # Step 6 - Prediction Layer
        predict = F.tanh(self.l1(aggre))
        predict = F.dropout(predict, p=self.dropout)
        return F.log_softmax(self.l2(predict))

    def _aggre(self, q_aware_reps, a_aware_reps):
        """
        Aggregation Layer handle
        Args:
            q_aware_reps - [bsz, q_len, 11*mp_dim+6]
            a_aware_reps - [bsz, a_len, 11*mp_dim+6]
        Return:
            size - [bsz, aggregation_lstm_dim*4]
        """
        _aggres = []
        q_output, _ = self.aggre_lstm(q_aware_reps)
        a_output, _ = self.aggre_lstm(a_aware_reps)

        f_q_output, b_q_output = q_output.split(self.aggregation_lstm_dim, q_output.dim()-1)
        f_a_output, b_a_output = a_output.split(self.aggregation_lstm_dim, a_output.dim()-1)

        # [bsz, aggregation_lstm_dim]
        _aggres.append(f_q_output[:, -1, :])
        _aggres.append(b_q_output[:, 0, :])
        _aggres.append(f_a_output[:, -1, :])
        _aggres.append(b_a_output[:, 0, :])
        return torch.cat(_aggres, dim=1)

    def _init_weights_and_bias(self, scope=0.1):
        """
        initialise weight and bias
        """
        self.word_emb.weight.data.uniform_(-scope, scope)
        self.word_lstm_cell.weight.data.uniform_(-scope, scope)
        self.context_lstm.weight.data.uniform_(-scope, scope)
        self.aggre_lstm.weight.data.uniform_(-scope, scope)
        self.l1.weight.data.uniform_(-scope, scope)
        self.l1.bias.data.fill_(0)
        self.l2.weight.data.uniform_(-scope, scope)
        self.l2.bias.data.fill_(0)

    def _init_corpus_embedding(self):
        """
        corpus embedding is a fixed vector for each individual corpus,
        which is pre-trained with word2vec
        """
        self.corpus_emb.weight.data.copy_(torch.from_numpy(self.corpora_emb))
        # frozen embedding parameters
        self.corpus_emb.weight.requires_grad = False

    def _word_repre_forward(self, input):
        """
        args:
            - input: q_words|a_words size: [bsz, sent_length, words_len]
                ps: q_words|a_words is matrix: corpus * words
        return:
            - output: [bsz, sent_length, word_lstm_dim]
        """
        # [bsz*sent_length, words_len]
        bsz = input.size(0)
        sent_length = input.size(1)
        input = input.view(-1, input.size(2))
        # [bsz*sent_length, words_len, word_emb_dim]
        encode = self.word_emb(input)
        # [bsz*sent_length ,words_len, word_emb_dim] -> [bsz*sent_length, words_len, word_lstm_dim]
        output, _ = self.word_lstm(encode)
        # [bsz*sent_length, words_len, word_lstm_dim] -> [bsz, sent_length, word_lstm_dim]
        output = output[:, -1, :].view(bsz, sent_length,
                                self.word_lstm_dim)
        return output

    def _word_repre_layer(self, input):
        """
        args:
            - input: (q_sentence, q_words)|(a_sentence, a_words)
              q_sentence - [bsz, sent_length]
              q_words - [bsz, sent_length, words_len]
        return:
            - output: [bsz, sent_length, context_dim]
        """
        sentence, words = input
        s_encode = self.corpus_emb(sentence)

        # [bsz, sent_length, word_emb_dim]
        w_encode = self._word_repre_forward(words)
        w_encode = F.dropout(w_encode, p=self.dropout, training=True, inplace=False)

        out = torch.cat((s_encode, w_encode), 2)
        return out

    def _context_repre_forward(self, input):
        """
        Implement: Context Representation Layer, Matching Layer

        Args:
            - input: q_context|a_context
              size: [bsz, sent_length, context_dim]]
        Return:
            - output: (f_output, b_output)
                size - ([bsz, sent_length, self.context_lstm_dim], [bsz, sent_length, self.context_lstm_dim])
        """
        output, _ = self.context_lstm(input)
        return output.split(self.context_lstm_dim, output.dim()-1)

    def _bilateral_match(self, cos_simi):
        """
        Args:
            - cos_simi - cosine similarity mask [bsz, answer_len, question_len]
        Return:
            q_aware_reps - [bsz, question_len, mp_dim*11+6]
            a_aware_reps - [bsz, answer_len, mp_dim*11+6]
        """
        # [bsz, q_len, a_len]
        cos_simi_q = cos_simi.permute(0, 2, 1)
        cos_simi_dim = cos_simi.dim()

        # [bsz, a_len, 1]
        q_aware_reps = [torch.max(cos_simi, cos_simi_dim-1, keepdim=True)[0],
                        torch.mean(cos_simi, cos_simi_dim-1, keepdim=True)]
        # [bsz, q_len, 1]
        a_aware_reps = [torch.max(cos_simi_q, cos_simi_dim-1, keepdim=True)[0],
                        torch.mean(cos_simi_q, cos_simi_dim-1, keepdim=True)]

        # Max Attentive Matching
        q_max_att = self._max_repres((self.q_repres, cos_simi))
        q_max_att_rep = self.pre_q_attmath_layer(self.a_repres, q_max_att)
        q_aware_reps.append(q_max_att_rep)

        a_max_att = self._max_repres((self.a_repres, cos_simi_q))
        a_max_att_rep = self.pre_a_attmath_layer(self.q_repres, a_max_att)
        a_aware_reps.append(a_max_att_rep)

        # Context MP Matching
        # range 1 - bilstm
        q_repr_context_f, q_repr_context_b = self._context_repre_forward(self.q_repres)
        a_repr_context_f, a_repr_context_b = self._context_repre_forward(self.a_repres)

        # range 2 - all match layers
        left_match = self._all_match_layer(a_repr_context_f,
                                           a_repr_context_b,
                                           q_repr_context_f,
                                           q_repr_context_b)
        right_match = self._all_match_layer(q_repr_context_f,
                                            q_repr_context_b,
                                            a_repr_context_f,
                                            a_repr_context_b)

        q_aware_reps.extend(left_match)
        a_aware_reps.extend(right_match)
        q_aware_reps = torch.cat(q_aware_reps, dim=2)
        a_aware_reps = torch.cat(a_aware_reps, dim=2)

        return q_aware_reps, a_aware_reps

    def _all_match_layer(self, repr_context_f, repr_context_b,
                        other_repr_context_f, other_repr_context_b):
        """
        Args:
            repr_context_f, repr_context_b|other_repr_context_f, other_repr_context_b - size: [bsz, this_len, context_lstm_dim], [bsz, other_len, context_lstm_dim]
        Return:
            List - size: [bsz, sentence_len, mp_dim] * 10*mp_dim+4
        """
        # pre sth before
        repr_context_f = repr_context_f.contiguous()
        repr_context_b = repr_context_b.contiguous()
        other_repr_context_f = other_repr_context_f.contiguous()
        other_repr_context_b = other_repr_context_b.contiguous()

        all_aware_repres = []
        this_cont_dim = repr_context_f.dim() # 3

        # [bsz, this_len, other_len]
        f_relevancy = F.cosine_similarity(other_repr_context_f.unsqueeze(1), repr_context_f.unsqueeze(2), dim=this_cont_dim)

        # [bsz, this_len, other_len]
        b_relevancy = F.cosine_similarity(other_repr_context_b.unsqueeze(1), repr_context_b.unsqueeze(2), dim=this_cont_dim)

        # first match - Full Match
        # gather the last time step of the forward|backward repres from the other sentences
        other_context_f_first = other_repr_context_f[:, -1, :]
        other_context_b_first = other_repr_context_b[:, 0, :]

        f_full_match = self.f_full_layer(repr_context_f, other_context_f_first)
        b_full_match = self.b_full_layer(repr_context_b, other_context_b_first)
        all_aware_repres.append(f_full_match)
        all_aware_repres.append(b_full_match)

        # second match - Maxpool Match
        f_max_match = self.f_max_layer(repr_context_f, other_repr_context_f)
        b_max_match = self.b_max_layer(repr_context_b, other_repr_context_b)
        all_aware_repres.append(f_max_match)
        all_aware_repres.append(b_max_match)

        # third match - Attentive Match
        f_att_cont = cosine_cont(other_repr_context_f, f_relevancy)
        f_att_repre = self.f_att_layer(repr_context_f, f_att_cont)
        b_att_cont = cosine_cont(other_repr_context_b, b_relevancy)
        b_att_repre = self.b_att_layer(repr_context_b, b_att_cont)
        all_aware_repres.append(f_att_repre)
        all_aware_repres.append(b_att_repre)

        # fourth match - Max Attentive Match
        f_max_att = self._max_repres((other_repr_context_f, f_relevancy))
        f_max_att_repres = self.f_max_att_layer(repr_context_f, f_max_att)
        b_max_att = self._max_repres((other_repr_context_b, b_relevancy))
        b_max_att_repres = self.b_max_att_layer(repr_context_b, b_max_att)
        all_aware_repres.append(f_max_att_repres)
        all_aware_repres.append(b_max_att_repres)

        # fifth - max & mean
        all_aware_repres.append(f_relevancy.max(2, keepdim=True)[0])
        all_aware_repres.append(f_relevancy.mean(2, keepdim=True))
        all_aware_repres.append(b_relevancy.max(2, keepdim=True)[0])
        all_aware_repres.append(b_relevancy.mean(2, keepdim=True))

        return all_aware_repres

    def _cosine_similarity_mask(self, simi):
        """
        Args:
            - simi: [bsz, answer_len, question_len]
        Returns:
            - size: [bsz, answer_len, question_len]
        PS:
            cosine_distance = (x1*x2 + y1*y2) / sqrt(x1**2 + y1**2) / sqrt(x2**2+ y2**2)
        """
        simi = torch.mul(simi, self.q_mask.unsqueeze(1).float())
        simi = torch.mul(simi, self.a_mask.unsqueeze(2).float())
        return simi

    def _max_repres(self, repre_cos):
        """
        Args:
            repre_cos - Tuple: (q_repres, cos_simi_q)|(a_repres, cos_simi)
                          Size: ([bsz, question_len, context_dim], [bsz, answer_len, question_len])| ...
        Return:
            size - [bsz, answer_len, context_dim] if question else [bsz, question_len, context_dim]
        """
        repres, cos_simi = repre_cos
        index = torch.max(cos_simi, 2)[1] # max_index
        return tf_gather(repres, index)
