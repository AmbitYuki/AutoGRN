import sys

sys.path.append('D:/Pycharm Project/GTS-change-master/model/pytorch/layers')
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.pytorch.cell import DCGRUCell
from model.pytorch.utils import scaled_Laplacian, cheb_polynomial
# from model.pytorch.FEDformer import Model
from model.pytorch.layers.Autoformer_EncDec import EncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from lib import utils
import pandas as pd
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from model.pytorch.copula import CorCopulaGCN
from model.pytorch.copula import CopulaModel

np.seterr(divide='ignore', invalid='ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        # self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.attention_unit = int(model_kwargs.get('attention_unit', 5))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel_1(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])
        # self.copuladcgru_layers = nn.ModuleList([CorCopulaGCN()])

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class EncoderModel_2(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class Attention_1(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.W1 = nn.Linear(self.hidden_state_size, self.attention_unit)
        self.W2 = nn.Linear(self.hidden_state_size, self.attention_unit)
        self.V = nn.Linear(self.attention_unit, 1)

    def forward(self, query, value):
        # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
        # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
        score = self.V(torch.tanh(self.W1(query) + self.W2(value)))

        attention_weights = F.softmax(score, dim=1)

        context_vector = attention_weights * value

        return context_vector, attention_weights


class Attention_2(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)

    def forward(self, query, key, value, mask=None, dropout=None):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))

        scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
                 / math.sqrt(query.size(-1))  # matmul是tensor的乘法-1为列的个数-2为行数

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class Attention_3(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        # self.hidden_dim=hidden_state_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmod = nn.Sigmoid()
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_state_size, self.hidden_state_size))
        self.dropout = nn.Dropout(0.1)
        # init.xavier_uniform(self.w_omega)

    def forward(self, input_Q, input_K, input_V):
        input_Q = torch.matmul(input_Q, self.w_omega)
        input_Q = self.tanh(input_Q)
        probs = torch.matmul(input_Q, input_K.transpose(1, 2)) / np.sqrt(input_V.shape[2])

        probs = self.softmax(probs)

        prob_arg = torch.argmax(probs, dim=-1)
        att_output = torch.matmul(probs, input_V)

        return att_output, probs, prob_arg


class Attention_4(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        # self.hidden_dim=hidden_state_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmod = nn.Sigmoid()
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_state_size, self.hidden_state_size))
        self.dropout = nn.Dropout(0.1)
        # init.xavier_uniform(self.w_omega)

    def forward(self, input_Q, input_K, input_V):
        # att_ouput = input_V.clone()
        att_input = torch.matmul(input_Q, self.w_omega)
        # att_input = self.tanh(att_input)
        probs = torch.matmul(att_input, input_K.transpose(1, 2))

        # probs=self.Entity_Attention_mask(output,input_entity_id)
        # probs=torch.matmul(torch.diag_embed(1/(torch.sum(probs,dim=-1)+1e-8)),probs)
        probs_softmax = self.softmax(probs)
        probs_sigmod = self.sigmod(probs)
        probs_sigmod_hat = probs_sigmod.gt(0.5).type_as(probs_softmax)
        # for i in range(probs.shape[0]):
        #     att_ouput[i] = torch.index_select(input_V[i], 0, probs[i])
        att_output_softmax = torch.matmul(probs_softmax, input_V)
        att_output_sigmod = torch.matmul(probs_sigmod_hat, input_V)
        # final_att_output = torch.reshape(att_ouput, [batch_size, max_length, hidden_dim])
        # final_att_output=self.dropout(att_ouput)
        att_output_softmax = att_output_softmax.squeeze(1)
        att_output_sigmod = att_output_sigmod.squeeze(1)
        probs_softmax = probs_softmax.squeeze(1)
        probs_sigmod = probs_sigmod.squeeze(1)

        return att_output_softmax, att_output_sigmod, probs_softmax, probs_sigmod


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, **model_kwargs):
        """
        Compute spatial attention scores
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Spatial_Attention_layer, self).__init__()
        self.num_of_vertices = int(model_kwargs.get('num_of_vertices'))
        self.num_of_features = int(model_kwargs.get('num_of_features'))
        self.num_of_timesteps = int(model_kwargs.get('num_of_timesteps'))

        global device
        self.W_1 = torch.randn(self.num_of_timesteps, requires_grad=True).to(device)
        self.W_2 = torch.randn(self.num_of_features, self.num_of_timesteps, requires_grad=True).to(device)
        self.W_3 = torch.randn(self.num_of_features, requires_grad=True).to(device)
        self.b_s = torch.randn(1, self.num_of_vertices, self.num_of_vertices, requires_grad=True).to(device)
        self.V_s = torch.randn(self.num_of_vertices, self.num_of_vertices, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

           initially, N == num_of_vertices (V)

        Returns
        ----------
        S_normalized: tensor, S', spatial attention scores
                      shape is (batch_size, N, N)

        """
        # get shape of input matrix x
        # batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # The shape of x could be different for different layer, especially the last two dimensions

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)

        # shape of rhs is (batch_size, T, V)
        # rhs = torch.matmul(self.W_3, x.transpose((2, 0, 3, 1)))
        rhs = torch.matmul(x.permute((2, 0, 1)), self.W_3)  # do we need to do transpose??
        # print(rhs)
        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s))

        # normalization
        S = S - torch.max(S, 1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return S_normalized


class Temporal_Attention_layer(nn.Module):
    """
    compute temporal attention scores
    """

    def __init__(self, **model_kwargs):
        """
        Temporal Attention Layer
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Temporal_Attention_layer, self).__init__()
        self.num_of_vertices = int(model_kwargs.get('num_of_vertices'))
        self.num_of_features = int(model_kwargs.get('num_of_features'))
        self.num_of_timesteps = int(model_kwargs.get('num_of_timesteps'))

        global device
        self.U_1 = torch.randn(self.num_of_vertices, requires_grad=True).to(device)
        self.U_2 = torch.randn(self.num_of_features, self.num_of_vertices, requires_grad=True).to(device)
        self.U_3 = torch.randn(self.num_of_features, requires_grad=True).to(device)
        self.b_e = torch.randn(self.num_of_features, self.num_of_timesteps, self.num_of_timesteps,
                               requires_grad=True).to(device)
        self.V_e = torch.randn(self.num_of_timesteps, self.num_of_timesteps, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, x^{(r - 1)}_h
                       shape is (batch_size, V, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: torch.tensor, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        """
        # _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # N == batch_size
        # V == num_of_vertices
        # C == num_of_features
        # T == num_of_timesteps

        # compute temporal attention scores
        # shape of lhs is (N, T, V)
        a = torch.matmul(x.permute(2, 1, 0), self.U_1)
        lhs = torch.matmul(torch.matmul(x.permute(2, 1, 0), self.U_1), self.U_2)

        # shape is (batch_size, V, T)
        # rhs = torch.matmul(self.U_3, x.transpose((2, 0, 1, 3)))
        rhs = torch.matmul(x.permute((2, 0, 1)), self.U_3)  # Is it ok to switch the position?

        product = torch.matmul(rhs, lhs.permute(1, 0))  # wd: (batch_size, T, T)

        # (batch_size, T, T)
        b = torch.sigmoid(product + self.b_e)
        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        # normailzation
        E = E - torch.max(E, 1, keepdim=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return E_normalized


class cheb_conv_with_SAt(nn.Module):
    """
    K-order chebyshev graph convolution with Spatial Attention scores
    """

    def __init__(self, **model_kwargs):
        """
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        """
        super(cheb_conv_with_SAt, self).__init__()
        self.num_of_filters = int(model_kwargs.get('num_of_filters'))
        self.K = int(model_kwargs.get('K'))
        self.num_of_features = int(model_kwargs.get('num_of_features'))
        self.num_of_timesteps = int(model_kwargs.get('num_of_timesteps'))

        global device
        self.Theta = torch.randn(self.K, self.num_of_timesteps, self.num_of_filters, requires_grad=True).to(device)

    def forward(self, x, adj_mx, spatial_attention):
        """
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        """
        # adj_mx = adj_mx.cpu()
        adj_mx = adj_mx.cpu().detach().numpy()
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = torch.tensor(cheb_polynomial(L_tilde, self.K), dtype=torch.float32).to(device)

        (seq_len, batch_size, num_sensor) = x.shape
        # (seq_len, batch_size, num_sensor * input_dim)

        # global device

        outputs = []
        # for s_step in range(seq_len):
        # shape is (batch_size, V, F)
        # graph_signal = x[:, :, :]
        output = torch.zeros(seq_len, batch_size, num_sensor).to(device)  # do we need to set require_grad=True?
        for k in range(self.K):
            # shape of T_k is (V, V)
            T_k = cheb_polynomials[k]
            # T_k = T_k

            # shape of T_k_with_at is (batch_size, V, V)
            T_k_with_at = T_k * (spatial_attention.reshape(-1, 5, 5))

            # shape of theta_k is (F, num_of_filters)
            # theta_k = self.Theta.data()[k]
            theta_k = self.Theta[k]

            # shape is (batch_size, V, F)
            # rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),  # why do we need to transpose?
            #                    graph_signal)
            rhs = torch.matmul((T_k_with_at.reshape(-1, 10, 10)).transpose(0, 2), x.transpose(0, 2))

            output = output + torch.matmul(rhs.permute(0, 2, 1), theta_k).permute(1, 2, 0)
            # outputs.append(output.expand_dims(-1))
        outputs.append(torch.unsqueeze(output, -1))

        return F.relu(torch.cat(outputs, dim=-1))


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])
        # self.copuladcgru_layers = nn.ModuleList([CorCopulaGCN()])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class GTSModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, temperature, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model_1 = EncoderModel_1(**model_kwargs)
        self.encoder_model_2 = EncoderModel_2(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.attention_layer_1 = Attention_1(**model_kwargs)
        self.attention_layer_2 = Attention_2(**model_kwargs)
        self.attention_layer_3 = Attention_3(**model_kwargs)
        self.attention_layer_4 = Attention_4(**model_kwargs)

        # self.SAt = Spatial_Attention_layer(**model_kwargs)
        # self.cheb_conv_SAt = cheb_conv_with_SAt(**model_kwargs)
        # self.TAt = Temporal_Attention_layer(**model_kwargs)

        # self.SAt = Spatial_Attention_layer(12, 64, 10)
        # self.cheb_conv_SAt = cheb_conv_with_SAt(num_of_filters=64, K=3, num_of_features=64)
        #
        # self.TAt = Temporal_Attention_layer(12, 64, 10)

        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.temperature = temperature
        self.dim_fc = int(model_kwargs.get('dim_fc', False))
        self.embedding_dim = 100
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.est_p_edge = float(model_kwargs.get('est_p_edge', 0.0))

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder_1(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model_1.seq_len):
            _, encoder_hidden_state = self.encoder_model_1(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def encoder_2(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model_2.seq_len):
            _, encoder_hidden_state = self.encoder_model_2(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    # def attention(self, encoder_query, encoder_value, encoder_key):
    #     encoder_hidden_state, _ = self.attention_layer(encoder_query, encoder_value, encoder_key)
    #     return encoder_hidden_state

    def attention_1(self, encoder_query, encoder_value):
        encoder_hidden_state, _ = self.attention_layer_1(encoder_query, encoder_value)
        return encoder_hidden_state

    #
    def attention_2(self, encoder_query, encoder_key, encoder_value):
        encoder_hidden_state, _ = self.attention_layer_2(encoder_query, encoder_key, encoder_value)
        return encoder_hidden_state

    def attention_3(self, encoder_query, encoder_key, encoder_value):
        encoder_hidden_state, _, _ = self.attention_layer_3(encoder_query, encoder_key, encoder_value)
        return encoder_hidden_state

    def attention_4(self, encoder_query, encoder_key, encoder_value):
        encoder_hidden_state, _, _, _ = self.attention_layer_4(encoder_query, encoder_key, encoder_value)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, node_feas, adj_dis, adj_knn, temp, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        # print(x)
        # class Configs(object):
        #     ab = 0
        #     modes = 32
        #     mode_select = 'random'
        #     # version = 'Fourier'
        #     version = 'Wavelets'
        #     moving_avg = [12, 24]
        #     L = 1
        #     base = 'legendre'
        #     cross_activation = 'tanh'
        #     seq_len = 96
        #     label_len = 48
        #     pred_len = 96
        #     output_attention = True
        #     enc_in = 10
        #     d_model = 28139
        #     embed = 'timeF'
        #     dropout = 0.05
        #     freq = 'h'
        #     factor = 1
        #     n_heads = 8
        #     d_ff = 16
        #     e_layers = 2
        #     d_layers = 1
        #     c_out = 7
        #     activation = 'gelu'
        #     wavelet = 0
        # configs = Configs()
        # model = Model(configs)

        # node_feas_FED = node_feas.unsqueeze(1)
        # node_feas_mark=torch.randn([a, b, 1])
        # node_feas=model.forward(node_feas_FED,node_feas_mark)

        # print(node_feas)  # [28139,10]
        # a,b= node_feas.size()
        model = EncoderLayer(d_model=10)
        node_feas_FED = node_feas.unsqueeze(0)  # [1,28139,10]
        node_feas_FED = model.forward(node_feas_FED).transpose(1, 0).transpose(2, 1)  # [1,28139,10]

        class SELayer(nn.Module):
            def __init__(self, channel, reduction=2):
                super(SELayer, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
                self.c = channel
                self.fc = nn.Sequential(
                    nn.Linear(self.c, self.c // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.c // reduction, self.c, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, l = x.size()
                # torch.Size([128, 96, 128])
                y = self.avg_pool(x)
                # print(y.shape)  # [128, 96, 1]
                y = y.view(b, c)
                # print(y.shape) # [128, 96]
                y = self.fc(y)
                # print(y.shape) # [128, 96]
                y = y.view(b, c, 1)
                return x * y.expand_as(x)

        node_feas_SE = node_feas.unsqueeze(2)
        # print(node_feas) #[28139,10,1]
        SE_net = SELayer(channel=10)
        node_feas_SE = SE_net(node_feas_SE)

        node_feas = node_feas_FED + node_feas_SE
        # print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))

        x = node_feas.transpose(1, 0).view(self.num_nodes, 1, -1)
        # print(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.hidden_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        # print(x.shape)

        ## gumbel adj
        adj = gumbel_softmax(x, temperature=temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        adj.masked_fill_(mask, 0)
        # print(adj)

        # spatial_At
        # spatial_At = self.SAt(inputs)
        # #
        # x_SAt = torch.matmul(inputs.reshape(1, -1, 12), spatial_At).reshape(12, 64, 10)

        # cheb gcn with spatial attention
        # (batch_size, num_of_vertices, num_of_vertices)
        # temporal_At = self.TAt(x_SAt)
        #
        # # (batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # spatial_gcn = self.cheb_conv_SAt(inputs, adj, temporal_At)
        # spatial_gcn = torch.squeeze(spatial_gcn)

        encoder_query = self.encoder_1(inputs, adj)
        encoder_value = self.encoder_2(inputs, adj_dis)
        # encoder_key = self.encoder_2(inputs, adj_dis)
        # encoder_hidden_state = self.attention_1(encoder_query, encoder_key)
        encoder_hidden_state1 = self.attention_1(encoder_query, encoder_value)
        encoder_hidden_state2 = self.attention_2(encoder_value, encoder_query, encoder_query)
        encoder_hidden_state3 = self.attention_3(encoder_value, encoder_query, encoder_query)
        encoder_hidden_state4 = self.attention_4(encoder_value, encoder_query, encoder_query)  # no
        # encoder_hidden_state = self.attention_2(encoder_qv, encoder_key)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_query, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, x.softmax(-1)[:, 0].clone().reshape(self.num_nodes, -1)
