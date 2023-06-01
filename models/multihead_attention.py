import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.args = args
        self.k = ((32*14 // args.patch_size) - 1) ** 2
        self.I = torch.eye(self.k).cuda()

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if self.args.activation == 'tanh':
            attention = torch.tanh(scores)
        elif self.args.activation == 'softmax':
            attention = torch.softmax(scores, dim=-1)
        elif self.args.activation == 'no':
            attention = scores
            
        if self.args.ignore_diag == 1:
            attention = self._ignore_diagonal(attention)
        return attention.matmul(value), attention

    def _ignore_diagonal(self, x):
        batch_size, x_size = x.shape[0], x.shape[1]
        I_batch = torch.stack([self.I] * batch_size, dim=0)
        return x * (1 - I_batch)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 args,
                 in_features,
                 head_num,
                 seq_len,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        self.args = args
        if (4 * in_features) % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.seq_len = seq_len

        self.attention_upsize = 4 * in_features

        if self.args.project_higher == 1:
            self.linear_q = nn.Sequential(nn.Linear(in_features, 2 * in_features, bias),
                                          nn.ReLU(),
                                          nn.Linear(2 * in_features, self.attention_upsize)
                                          )
            self.linear_k = nn.Sequential(nn.Linear(in_features, 2 * in_features, bias),
                                          nn.ReLU(),
                                          nn.Linear(2 * in_features, self.attention_upsize)
                                          )
            self.linear_v = nn.Sequential(nn.Linear(in_features, 2 * in_features, bias),
                                          nn.ReLU(),
                                          nn.Linear(2 * in_features, self.attention_upsize)
                                          )
            self.linear_o = nn.Sequential(nn.Linear(self.attention_upsize, in_features, bias))
        else:
            self.linear_q = nn.Sequential(nn.Linear(in_features, in_features, bias))
            self.linear_k = nn.Sequential(nn.Linear(in_features, in_features, bias))
            self.linear_v = nn.Sequential(nn.Linear(in_features, in_features, bias))
            self.linear_o = nn.Sequential(nn.Linear(in_features, in_features, bias))


    def forward(self, query, key, value, mask=None):
        query, key = self.linear_q(query), self.linear_k(key)
        value = self.linear_v(value)

        if self.activation is not None:
            query = self.activation(query)
            key = self.activation(key)
            # value = self.activation(value)

        query = self._reshape_to_batches(query)
        key = self._reshape_to_batches(key)
        value = self._reshape_to_batches(value)

        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attention = ScaledDotProductAttention(self.args)(query, key, value, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attention

    def compute_attention(self, query, key, mask=None):
        query, key = self.linear_q(query), self.linear_k(key)

        query = self._reshape_to_batches(query)
        key = self._reshape_to_batches(key)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)

        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.tanh(scores)

        attention = attention.reshape(-1, self.head_num, self.seq_len, self.seq_len)
        # attention = torch.mean(attention, dim=1)
        return attention

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
