import torch
import torch.nn as nn
import torch.nn.functional as F

# Q = torch.randn(16,28,300)
# K = torch.randn(16,28,300)
# V = K
#
# # 获得注意力权重
# alpha_mat = torch.matmul(Q, K.transpose(1,2))              #. [16,28,28]
# # 归一化注意力权重,alpha把第一求和，softmax把第二位归一化
# alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)   # [16,1,28]
# # 进行加权和
# x = torch.matmul(alpha, V).squeeze(1)                     # [16,300]

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Attention_2(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.Q = nn.Linear(self.hidden_state_size, self.attention_unit)
        self.K = nn.Linear(self.hidden_state_size, self.attention_unit)
        self.V = nn.Linear(self.attention_unit, 1)

        def forward(self, query, key):
            alpha_mat = torch.matmul(self.Q(query), self.K(key).transpose(1, 2))  # . [16,28,28]
            # 归一化注意力权重,alpha把第一求和，softmax把第二位归一化
            alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)  # [16,1,28]
            # 进行加权和
            x = torch.matmul(alpha, self.V).squeeze(1)  # [16,300]

            return x



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


class Attention2(nn.Module):
    def __init__(self,**model_kwargs):
        super(Attention2, self).__init__()
        self.hidden_dim=hidden_state_size
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=-1)
        self.sigmod = nn.Sigmoid()
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_dim , self.hidden_dim))
        self.dropout=nn.Dropout(0.1)
        # init.xavier_uniform(self.w_omega)

    def masked_attention_softmax(self,atten_scores, src_mask, src_length_masking=True):
        if src_length_masking:
            # compute masks
            bsz, max_src_len,_ = atten_scores.size()
            src_mask=src_mask.unsqueeze(1).expand(bsz,max_src_len,_).type_as(atten_scores)
            # Fill pad positions with -inf
            scores = atten_scores.masked_fill(src_mask == 0,-np.inf)
            #scores=atten_scores*src_mask
        # Cast to float and then back again to prevent loss explosion under fp16.
        return scores

    def forward(self,input_Q,input_K,input_V,mask):
        #att_ouput = input_V.clone()
        att_input = torch.matmul(input_Q, self.w_omega)
        #att_input = self.tanh(att_input)
        att_scores = torch.matmul(att_input, input_K.transpose(1, 2))

        probs = self.masked_attention_softmax(
            att_scores, mask, True
        )
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