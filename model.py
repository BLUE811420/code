import datetime
import math
import pdb

from recbole.model.layers import TransformerEncoder

from Mamba import Model
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import GlobalAggregator
from torch.nn import Module, Parameter

import torch
import torch.nn.functional as F


class AWLoss(torch.nn.Module):
    def __init__(self, node,gamma=6.0):
        super(AWLoss, self).__init__()
        self.gamma = gamma
        self.node = node

    def forward(self, logits, targets):
        """
        logits: Tensor of model outputs (before softmax) with shape [batch_size, num_classes]
        targets: Tensor of ground truth labels with shape [batch_size, num_classes] (one-hot encoded)
        """
        # Compute probabilities
        probabilities = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.node-1)
        # Select the probabilities corresponding to the labels
        true_probs = torch.sum(probabilities * targets_one_hot, dim=1)

        # Compute the modulating factor
        modulating_factor = (2 - 2 * true_probs) ** self.gamma

        # Compute the log of the probabilities
        log_p = torch.log(true_probs + 1e-6)  # Adding a small value to prevent numerical instability

        # Calculate the final loss
        loss = -torch.mean(modulating_factor * log_p)

        return loss




class TransNet(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.n_layers = 2
        self.n_heads = opt.n_heads
        self.hidden_size = opt.hiddenSize
        self.inner_size = 256
        self.hidden_dropout_prob = 0.5
        self.attn_dropout_prob = 0.5
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02
        self.opt = opt
        #self.mamba = Mambalayer(self.hidden_size,self.opt)
        self.position_embedding = nn.Embedding(300, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        # 如果序列是倒序的，反转position_ids
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        if self.opt.dataset == 'Tmall':
            input_emb = item_emb + position_embedding
        else:
            input_emb = item_emb + position_embedding
        #input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        #output = self.mamba(input_emb)

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

import torch
import torch.nn as nn
import math

class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusion, self).__init__()
        self.feature_dim = feature_dim
        self.gate_weight = nn.Parameter(torch.Tensor(2, feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

    def forward(self, x, y):
        # 平均每个时间步的特征
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)
        combined_mean = torch.cat((x_mean, y_mean), dim=-1)

        # 计算全局门控信号
        gate_signal = torch.sigmoid(torch.matmul(combined_mean, self.gate_weight))
        gate_signal = gate_signal.expand_as(x)  # 扩展门控信号匹配原始输入维度

        # 应用门控信号
        z = gate_signal * x + (1 - gate_signal) * y
        return z



# 魔心实现核心代码
class CombineGraph(Module):
    def __init__(self, opt, num_node,all_node_neighbor,mask):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.beta = opt.beta
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.num_node = num_node
        # 隐层输出纬度
        self.dim = opt.hiddenSize
        # 局部和全局dropout系数
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        # 迭代次数
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.max_neighbor = opt.all_neighbor
        self.gate = GatedFusion(self.dim)
        # 节点邻居，下标是itemId，对应值是邻居item的集合
        # self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.all_neighbor = trans_to_cuda(torch.Tensor(all_node_neighbor)).long()
        self.all_neighbor_mask = mask
        # # self.adj_pre_node = trans_to_cuda(torch.Tensor(adj_pre_node)).long()
        # # 节点邻居出现的次数，下标是itemId，对应值是adj_all中邻居item的出现次数
        # self.num = trans_to_cuda(torch.Tensor(num)).float()
        # self.num_pre_node = trans_to_cuda(torch.Tensor(num_pre_node)).float()
        self.net = TransNet(self.opt)
        # self.local_gnn = GNN(self.dim,opt)
        # 会话级信息聚合器，只有一层
        # self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        # 全局级信息聚合器，有self.hop个
        self.global_agg = []
        # self.global_agg_pre = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn,self.opt, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn,self.opt, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        # item节点的embedding表示
        self.embedding = nn.Embedding(num_node, self.dim, padding_idx=0)

        # 反向位置信息的embedding表示，但是这里为什么限定是200？
        self.pos_embedding = nn.Embedding(300, self.dim)
        self.select_num = 50
        # Parameters
        self.w1 = nn.Parameter(torch.Tensor(self.dim, 2))
        self.w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_4 = self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        # self.w_4 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.dropout = nn.Dropout(0.2)
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim)
        self.glu4 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_one = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_two = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_three = nn.Linear(self.dim, 1, bias=False)
        self.linear_four = nn.Linear(3 * self.dim, self.dim)
        self.linear_t = nn.Linear(self.dim, self.dim, bias=False)  # target attention
        self.glu1_select = nn.Linear(self.dim, self.dim)
        self.glu2_select = nn.Linear(self.dim, self.dim, bias=False)
        self.w2_select = nn.Parameter(torch.Tensor(self.dim, 1))
        self.temperature = opt.temperature
        self.last_layernorm = torch.nn.LayerNorm(self.dim, eps=1e-8)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-8)
        self.emb_dropout = torch.nn.Dropout(p=0.2)
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.awloss = AWLoss(self.num_node)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            param.data.uniform_(-stdv, stdv)
        self.embedding.weight.data[0] = 0

    def triplet_loss(self,anchor, positive, negative, margin=1.0):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(distance_positive - distance_negative + margin)
        return losses.mean()

    # 假设 long_term_rep, short_term_rep, long_term_out, short_term_out 已经定义好
    # 它们的维度都是 [B, L, D]

    def sample(self, target, n_sample):
        """
        返回target对应的邻居以及邻居出现的次数
        :param target: 目标item
        :param n_sample: 采样个数
        :return: 目标item的邻居，以及邻居出现的次数
        """
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        # target.view(-1) 将target展平
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def sample_all_adj(self, target):
        """
        返回target对应的邻居以及邻居出现的次数
        :param target: 目标item
        :param n_sample: 采样个数
        :return: 目标item的邻居，以及邻居出现的次数
        """
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        # target.view(-1) 将target展平
        return self.all_neighbor[target.view(-1)]

    def select_topk(self, item_embedding_ave,sess_embedding, mask,pos=True):
        # zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        length = sess_embedding.size(1)
        # sess_item_embed = torch.cuda.FloatTensor(self.batch_size, len, self.emb_size).fill_(0)
        # item_embedding = torch.cat([zeros, item_embedding], 0)#[node,dim]
        # get_sess_item = lambda i: item_embedding[reversed_sess_item[i]]
        # for i in torch.arange(self.batch_size):#batch_size
        #     sess_item_embed[i] = get_sess_item(i)#[batch,len,dim]
        item_embedding = self.embedding.weight[1:]#[node,dim]
        if(pos == True):
            pos_emb = self.pos_embedding.weight[:length]#[batch_max_session_len,dim]
            beta_sess_self = pos_emb+sess_embedding
        else:
            beta_sess_self = sess_embedding
        mask = mask.float().unsqueeze(-1)#[batch,len,1]
        #sess_mean = torch.unsqueeze(torch.div(torch.sum(sess_item_embed,1),session_len),1).repeat(1,len,1)#[batch,1,dim]
        sess_mean = item_embedding_ave
        #sess_mean = sess_mean.squeeze(1)
        #sess_mean = F.normalize(sess_mean, dim=-1, p=2)
        beta_sess = torch.sigmoid(self.glu1_select(beta_sess_self) + self.glu2_select(sess_mean))
        beta_sess = torch.matmul(beta_sess, self.w2_select)#[batch,len,1]
        beta = beta_sess * mask#[batch,len,1]
        softatt_sess = torch.sum(beta * sess_embedding, 1)
        softatt_sess = 7 * F.normalize(softatt_sess, dim=-1, p=2)#[batch,dim]
        item_embedding = F.normalize(item_embedding, dim=-1, p=2)
        score = torch.matmul(softatt_sess, item_embedding.transpose(0, 1))
        value, index = score.topk(self.select_num, dim=1, largest=True, sorted=True)#[batch,topk]
        return value,index,score
    # 得到最后的item表示并取出堆叠成会话，hidden的维度为：batch_size*seq_len*embedding_size（会话数*item数*embedding维度）。

    def generate_session(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_3)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu3(nh) + self.glu4(hs))
        beta = torch.matmul(nh, self.w_4)
        beta = beta * mask
        select = torch.mul(beta, hidden)
        return select

    def generate_session_no_position(self, hidden, hidden1,item,session_embedding,mask,targets):
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = hidden
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        hidden1 = self.gate(select, hidden1)
        hidden1 = F.normalize(hidden1, dim=-1)
        embedding = self.embedding.weight[1:]
        embedding = self.emb_dropout(embedding)
        embedding = F.normalize(embedding, dim=-1)
        score = torch.matmul(hidden1, embedding.transpose(1, 0)) / self.temperature
        return score

    def compute_scores(self, hidden, hidden1,item,session_embedding,mask,targets):
        # inner_mask = torch.zeros(hidden.shape[0],self.num_node, dtype=bool, device='cuda')
        # inner_mask = inner_mask.scatter(1, item, 1)
        # out_mask = ~inner_mask
        # inner_mask = inner_mask[:, 1:]
        # out_mask = out_mask[:, 1:]
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1) # b d
        #out = torch.cat([select, hidden1], -1)
        hidden1 = self.gate(select, hidden1)
        hidden1 = F.normalize(hidden1, dim=-1)
        embedding = self.embedding.weight[1:]
        embedding = self.emb_dropout(embedding)
        embedding = F.normalize(embedding, dim=-1)
        score = torch.matmul(hidden1, embedding.transpose(1, 0)) / self.temperature
        return score

    def loss_target(self,score,index,tar):
        score_all = torch.zeros(score.shape[0], self.num_node, dtype=torch.float32, device='cuda')
        for i in torch.arange(score_all.shape[0]):
            score_all[i][index[i]] = score[i]
        loss = self.loss_function(score_all,tar)
        return loss,score_all
        # score_all = torch.zeros(self.batch_size, self.num_node, dtype=torch.float32, device='cuda')
        # # Create row indices ranging from 0 to batch_size-1
        # row_indices = torch.arange(score_all.shape[0], device='cuda')
        # # Use advanced indexing to place 'score' at 'index' positions in 'score_all'
        # score_all[row_indices, index] = score
        # # Calculate the loss
        # loss = self.loss_function(score_all, tar)
        # return loss, score_all


    def compute_scores_test(self, hidden, mask,item_embedding_ave,session_embedding,targets,hidden1,sum_hidden):
        targets = trans_to_cuda(targets).long()
        value,index,score = self.select_topk(item_embedding_ave, session_embedding,mask,pos=True)
        loss_select = self.loss_function(score,targets-1)
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask,1) - 1]  # batch_size x latent_size#每个序列中最后一个元素h_local
        # q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size b 1 d
        # q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size b l d
        # alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (b,s,1)
        # # alpha = torch.sigmoid(alpha) # B,S,1
        # alpha = F.softmax(alpha, 1)  # B,S,1
        a = self.generate_session(hidden, mask) # B,D
        # 仅使用 top k 候选集
        b = self.embedding.weight[1:][index.flatten()]  # 使用 flatten 是为了处理多维索引的问题

        b = b.view(hidden.shape[0], -1, b.shape[-1])  # 重塑以匹配 batch_size
        sum_hidden = a + sum_hidden
        # 调整目标商品嵌入的形状以便进行矩阵乘法
        target_item_embeddings = b.transpose(1, 2)  # [batch_size, embedding_dim, n]

        # 计算注意力分数
        attention_scores = torch.bmm(sum_hidden, target_item_embeddings)  # [batch_size, seq_length, n]
        attention_scores = F.softmax(attention_scores, dim=1)  # 在序列长度上进行softmax

        # 使用注意力分数聚合用户历史
        context_vectors = torch.bmm(sum_hidden.transpose(1, 2),attention_scores)  # [batch_size, embedding_dim, n]
        context_vectors = context_vectors.transpose(1, 2)  # [batch_size, n, embedding_dim]
        context_vectors = F.normalize(context_vectors, dim=-1)
        # 应用 target attention
        #hidden_masked = sum_hidden * mask.view(mask.shape[0], -1, 1).float()
        # qt = self.linear_t(hidden_masked)
        # beta = F.softmax(torch.bmm(qt, b.transpose(1, 2)), dim=-1)
        # beta = beta.transpose(1, 2)
        # target = torch.bmm(beta, hidden_masked)
        # # target = F.normalize(target, dim=-1)
        # # hidden1 = torch.cat([hidden1.view(ht.shape[0], 1, ht.shape[1]),target,a.view(ht.shape[0], 1, ht.shape[1])],-1)
        # # hidden1 = self.linear_four(hidden1)
        # hidden1 = hidden1.view(ht.shape[0], 1, ht.shape[1]) + target + a.view(ht.shape[0], 1, ht.shape[1])
        # hidden1 = F.normalize(hidden1, dim=-1)
        # 合并信息并计算得分
        b = F.normalize(b, dim=-1)
        scores = torch.sum(context_vectors * b,dim=-1)
        loss_item, scores_item = self.loss_target(scores, index, targets - 1)

        return 1.0 * loss_select, 1.0 * loss_item, scores_item


        # forward函数，输入会话，邻居项目矩阵，mask， item。item是会话中不重复的item列表（已补0）；
    # 但这里为什么还需要一个未去重的inputs会话序列？这里没有采用类似SR-GNN的邻接矩阵A的计算方式（SR-GNN中的邻接矩阵A就是在原始序列的基础上来的），所以需要传入原始序列进行计算
    def forward(self, inputs, mask_item, item, alias_inputs, neighbor,neighbor_mask,count):
        """
        计算会话的embedding
        :param inputs: 去重后的会话序列，batch_size*seq_len
        :param adj: item邻居矩阵，并不是邻接矩阵
        :param mask_item: 会话序列掩码,batch_size*seq_len
        :param item: 原始会话序列，batch_size*seq_len，已补0
        :return: 会话的embedding
        """
        # 批处理大小
        batch_size = inputs.shape[0]
        # 序列长度
        seqs_len = inputs.shape[1]

        # 输入去重之后的序列，得到item的embedding
        # h_all_neighbor = self.embedding(self.all_neighbor)
        x = self.embedding(item)
        x = self.dropout(x)
        alpha = self.net(item, x)
        # print(alpha.shape)
        hidden = torch.mul(alpha, x)
        seq_output1 = torch.sum(alpha * x, dim=1)
        #h_long = alpha * x
        # seq_output1 = F.normalize(seq_output1, dim=-1)
        # seq_output = F.normalize(seq_output, dim=-1)
        # print(seq_output.shape)

        neighbor_embedding = self.embedding(neighbor) * neighbor_mask.float().unsqueeze(-1)
        #neighbor_embedding = self.emb_dropout(neighbor_embedding)


        item_neighbors = [inputs]
        item_neighbors1 = [item]
        weight_neighbors = []
        # item_pre_neighbors = [inputs]
        # weight_pre_neighbors = []
        support_size = seqs_len

        all_neighbor = self.sample_all_adj(item)
        all_neighbor_embeddings = self.embedding(all_neighbor)
        #print(neighbor_embeddings.shape)
        all_neighbor_mask = all_neighbor != 0
        masked_embeddings = all_neighbor_embeddings * all_neighbor_mask.unsqueeze(-1).float()  # 应用mask
        neighbor_count = all_neighbor_mask.sum(dim=-1, keepdim=True)  # 计算每个序列位置的有效邻居数
        neighbor_count = neighbor_count.clamp(min=1)  # 防止除以0
        average_embeddings = masked_embeddings.sum(dim=1) / neighbor_count.float()

        # 将结果reshape回 [batch_size, seq_length, embedding_dim]
        long_term_p = average_embeddings.view(batch_size, seqs_len, -1)# 每个item的长期代表
        last_nonzero_idx = torch.sum(neighbor_mask, dim=2) - 1
        # Step 2: 创建索引张量
        # Step 3: 使用索引和最后一个不为零的位置的索引来选择 embedding
        default_idx = torch.zeros_like(last_nonzero_idx)  # 假设默认索引为零
        last_nonzero_idx = torch.where(torch.sum(neighbor_mask, dim=2) == 0, default_idx, last_nonzero_idx)
        last_nonzero_idx = torch.tensor(last_nonzero_idx, dtype=torch.long)
        short_term_p = torch.gather(neighbor_embedding, 2, last_nonzero_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1,neighbor_embedding.shape[-1]))
        short_term_p = short_term_p.squeeze(2)

        # self.hop是迭代次数，以1次为例（代码默认值为1）
        # 取出会话包含的item的所有邻居及其出现次数记为item_sample_i,weight_sample_i
        # for i in range(1, self.hop + 1):
        #     # ======== 断点调试 =======
        #     # pdb.set_trace()
        #     # ======== 断点调试 =======
        #     item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
        #     # item_sample_pre_i, weight_sample_pre_i = self.sample_pre(item_neighbors[-1], self.sample_num)
        #     support_size *= self.sample_num
        #     item_neighbors.append(item_sample_i.view(batch_size, support_size))
        #     # item_pre_neighbors.append(item_sample_pre_i.view(batch_size, support_size))
        #     weight_neighbors.append(weight_sample_i.view(batch_size, support_size))
            # weight_pre_neighbors.append(weight_sample_pre_i.view(batch_size, support_size))
        # for循环结束，item_neighbors就包含两项（因为hop默认是1），一项是batch的会话序列，一项是这些会话序列对应的邻居，取出它们的embedding

        # item_neighbors包含两项（因为hop默认是1），一项是batch的会话序列，一项是这些会话序列对应的邻居，取出它们的embedding；
        # 但这里会再一次生成inputs的embedding，因为item_neighbors[0]就是inputs，和前面的`item_neighbors = [inputs]`重复.
        # entity_vectors[0].shape：batch_size*seq_len*hidden_size；entity_vectors[1].shape：batch_size*(seq_len*sample_num)*hidden_size;
        entity_vectors = [self.embedding(i) for i in item_neighbors1]# 当前原始会话表征
        # entity_vectors = self.dropout
        # print(entity_vectors[1].shape)
        # short_item_sample_i, _ = self.sample(item_neighbors1[-1], self.sample_num)
        # # support_size *= self.sample_num
        # item_neighbors1.append(short_item_sample_i.view(batch_size, seqs_len, self.sample_num))
        # # print(item_neighbors1[1])
        # short_term = neighbor
        # # 假设 short_term 已经定义，并且包含非零和零值
        # batch_size, seqs_len, sample_num = short_term.shape
        # # 创建非零掩码
        # nonzero_mask = short_term != 0  # 假设非零邻居不等于零
        # # 生成每个元素的索引
        # indices = torch.arange(self.sample_num, device=short_term.device).expand(batch_size, seqs_len, self.sample_num)
        # # 计算每个元素的最后一个非零元素的索引
        # last_indices = (nonzero_mask.long() * indices).max(dim=2, keepdim=True).values
        # # 使用 last_indices 来从 short_term 中提取最后一个非零邻居
        # last_nonzero_neighbors = torch.gather(short_term, 2, last_indices)
        # short_term_p = self.embedding(last_nonzero_neighbors).squeeze(2)
        # print(last_nonzero_neighbors.shape)
        # last_nonzero_neighbors 现在包含每个序列中每个元素的最后一个非零邻居

        # entity_pre_vectors = [self.embedding(i) for i in item_pre_neighbors]
        # print(entity_vectors[0])
        # print(entity_vectors[1])
        # 本身序列+邻居序列（1hop或者1hop+2hop）的embedding
        # weight_vectors[0]的shape为：batch_size*(seq_len*sample_num)
        weight_vectors = weight_neighbors
        # weight_pre_vectors = weight_pre_neighbors
        session_info = []
        # session_pre_info = []
        # 进行掩码操作，使得item_emb中补0的值对应的embedding也全为0
        # mask_item的shape为：batch_size*seq_len；mask_item.float().unsqueeze(-1)的shape为：batch_size*seq_len*1；item_emb的shape为：batch_size*seq_len*self.dim
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        #item_emb = self.dropout(item_emb)

        # mean，会话序列中所有项目的均值，对应公式3的操作，用于后续公式2中的s
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # ======== 断点调试 =======
        # pdb.set_trace()
        # ======== 断点调试 =======

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        # sum_item_emb的shape为：batch_size*self.dim；sum_item_emb.unsqueeze(-2).shape为：batch_size*1*self.dim
        sum_item_emb = sum_item_emb.unsqueeze(-2)# 会话的平均embedding表征
        for i in range(self.hop):
            # 实际执行 session_info.append(sum_item_emb.repeat(1, entity_vectors[0].shape[1], 1))；
            # session_info添加了一个大小为batch_size*seq_len*self.dim的tensor
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
            # session_pre_info.append(sum_item_emb.repeat(1, entity_pre_vectors[i].shape[1], 1))
            # 为后面的计算做准备，将sum_item_emb的shape扩展为batch_size*seq_len*hidden_size
        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                # 提取hop对应的聚合函数self.global_agg[n_hop]
                aggregator = self.global_agg[n_hop]
                # aggregator_pre = self.global_agg_pre[n_hop]
                # self_vectors传入的是entity_vectors[0]，batch_size*seq_len*self.dim，输入序列的embedding
                vector = aggregator(self_vectors=item_emb, # 输入序列的embedding
                                    # neighbor_vector是输入序列的邻居结点对应的embedding，shape为：batch_size*seq_len*sample_num*self.dim
                                    neighbor_vector=neighbor_embedding,
                                    masks=neighbor_mask,
                                    batch_size=batch_size,
                                    # weight_vectors[0]的shape为：batch_size*(seq_len*sample_num)，传入的neighbor_weight的shape为batch_size*seq_len*sample_num
                                    neighbor_weight=count,
                                    # 传入的extra_vector为batch_size*seq_len*self.dim的会话序列均值，对应公式2中的s
                                    extra_vector=session_info[hop]) # 会话的平均embedding表征
                # 将当前hop学习到的表示保存下来，作为下一个hop的输入
                entity_vectors_next_iter.append(vector)
                # entity_pre_vectors_next_iter.append(vector_pre)
            entity_vectors = entity_vectors_next_iter
            # entity_pre_vectors = entity_pre_vectors_next_iter
        # 取出最后一次更新得到的表示（因为entity_vectors中只包含最后一次更新的表示，因此下标是0）
        h_global = entity_vectors[0]
        #sum_hidden = h_global + hidden
        # h_global = self.dropout(h_global)
        # h_global = self.mamba(h_global)
        # h_global = F.normalize(h_global, p=2, dim=-1)
        # h_pre_global = entity_pre_vectors[0].view(batch_size, seqs_len, self.dim)
        # combine
        # 通过dropout，再进行sum pooling，从而得到最终的item表示output。


        # h_hidden = h_local + h_global
        # mamba_output = self.mamba(h_for_mamba)
        # 输出节点的最终表示，大小为batch_size*seq_len*hidden_size
        # get = lambda index: h_global[index][alias_inputs[index]]
        # 获取会话序列的初始embedding，大小为batch_size*seq_len*hidden_size
        # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

        #计算长期输出和长短期代表之间的损失
        loss_long_term = self.triplet_loss(hidden, long_term_p, short_term_p)
        #loss_long_term1 = self.triplet_loss(long_term_p, hidden, short_term_p)

        # 计算短期输出和长短期代表之间的损失
        loss_short_term = self.triplet_loss(h_global, short_term_p, long_term_p)
        #loss_short_term1 = self.triplet_loss(short_term_p,h_global, long_term_p)

        # 总损失
        total_loss = loss_long_term + loss_short_term


        #output = self.gate(seq_hidden, hidden)
        return h_global, seq_output1, item,sum_item_emb,item_emb,hidden,0.1 * total_loss




def trans_to_cuda(variable):
    # ======== 内存使用信息摘要打印 =======
    # CUDA out of memory.
    torch.cuda.memory_summary(device=None, abbreviated=False)
    # ======== 内存使用信息摘要打印 =======
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, use_item_cl_loss):
    # 这里data返回的依次是：alias_inputs、adj、items、mask、target、u_input
    alias_inputs,items, mask, targets,inputs,neighbor,neighbor_mask,count = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    # adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    neighbor = trans_to_cuda(neighbor).long()
    neighbor_mask = trans_to_cuda(neighbor_mask).long()
    count = trans_to_cuda(count).long()
    #targets = trans_to_cuda(targets).long()
    #inputs是原始的，item是去重之后的
    # 获得item的表示，这里应该是传入 inputs, adj, mask, items 的顺序？
    seq_hidden,seq_hidden1,item,item_embedding_ave,item_emb,sum_hidden,loss1 = model(items, mask, inputs, alias_inputs,neighbor,neighbor_mask,count)# items是去重之后的，inputs是原始的

    # ======== 断点调试 =======
    # pdb.set_trace()
    # ======== 断点调试 =======
    if model.opt.dataset == 'Tmall':
        score = model.generate_session_no_position(seq_hidden,seq_hidden1,item,item_emb, mask,targets)
    else:
        score = model.compute_scores(seq_hidden,seq_hidden1,item,item_emb, mask,targets)
    # a = model.generate_session(seq_hidden,seq_hidden1,mask)
    #loss1, loss2, score = model.compute_scores_test(seq_hidden, mask,item_embedding_ave,item_emb,targets,seq_hidden1,sum_hidden)

    return targets, score, loss1


def train_test(model, train_data, test_data, cl_loss_function, use_item_cl_loss=False, sampled_item_size=100, temperature=0.1, item_cl_loss_weight=0.1):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    # train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
    #                                            shuffle=True, pin_memory=True)
    # ======== 修改 ==========
    # 修改num_workers为2，太大了本地没法跑
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=3, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    # ======== 修改 ==========
    for data in tqdm(train_loader):
        # 这里data返回的依次是：alias_inputs、adj、items、mask、target、u_input
        model.optimizer.zero_grad()
        targets, scores, loss1 = forward(model, data,use_item_cl_loss)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + loss1
        #loss = loss1 + loss2
        #loss = loss + loss1
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=True, pin_memory=True)
    result = []
    hit_20, mrr_20, hit_10, mrr_10 = [], [], [], []
    for data in test_loader:
        # targets的shape为(batch_size-1,)给定的目标值；scores的shape为batch_size*item_num（数据集中item的总个数），scores的内容是每个会话对应的候选项的预测分数
        targets, scores, _ = forward(model, data, use_item_cl_loss)
        # 返回候选项中分数最高的前20个item的索引（其实就是itemId-1），这里索引是item生成embedding时的数组索引，所以索引和itemId差值为1，即index=itemId-1，所以计算hit时需要target-1
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_10 = scores.topk(10)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        targets = targets.numpy()
        for score_20, score_10, target, mask in zip(sub_scores_20, sub_scores_10, targets, test_data.mask):
            # 计算hit@20和hit@10
            hit_20.append(np.isin(target - 1, score_20))
            hit_10.append(np.isin(target - 1, score_10))
            # 计算mrr@20和mrr@10
            if len(np.where(score_20 == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score_20 == target - 1)[0][0] + 1))

            if len(np.where(score_10 == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score_10 == target - 1)[0][0] + 1))

    # 计算平均hit和mrr指标
    result.extend([np.mean(hit_20) * 100, np.mean(mrr_20) * 100, np.mean(hit_10) * 100, np.mean(mrr_10) * 100])

    return result
