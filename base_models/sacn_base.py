import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models.layers.wgcn_layer import WGCNLayer


class SACNBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 dim,
                 num_layer,
                 num_channel,
                 kernel_length,
                 dropout=0.0,
                 ):
        super(SACNBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.dim = dim
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.kernel_length = kernel_length
        self.drop_prop = dropout

        self.encoder = WGCNEncoder(self.num_relation, self.dim, self.num_layer)
        self.decoder = ConvTransEDecoder(self.dim, self.num_channel, self.kernel_length, self.drop_prop)

        self.entity_embed = torch.nn.Embedding(self.num_entity, self.dim)
        self.relation_embed = torch.nn.Embedding(self.num_relation, self.dim)

    def forward(self, h, query):
        score = self.decoder(h, self.relation_embed.weight, query)
        return score


class ConvTransEDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_channel,
                 kernel_length,
                 dropout=0.0):
        super(ConvTransEDecoder, self).__init__()
        self.input_dim = input_dim
        self.c = num_channel
        self.k = kernel_length

        self.block0 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2),
            torch.nn.Dropout(dropout)
        )

        self.block1 = torch.nn.Sequential()
        if kernel_length % 2 != 0:
            self.block1.append(nn.ZeroPad2d((int(kernel_length / 2), int(kernel_length / 2), 0, 0)))
        else:
            self.block1.append(nn.ZeroPad2d((int(kernel_length / 2) - 1, int(kernel_length / 2), 0, 0)))
        self.block1.append(nn.Conv2d(1, num_channel, (2, kernel_length)))

        self.block2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.c),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
            nn.Linear(input_dim * num_channel, input_dim, bias=False),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.ReLU()
        )

    @staticmethod
    def _apply_seq(seq_module: nn.Sequential, tensor: torch.Tensor):
        """
        Apply sequential module safely when batch size is 1 by running BatchNorm in eval mode.
        """
        for layer in seq_module:
            if isinstance(layer, nn.BatchNorm1d) and tensor.shape[0] == 1 and layer.training:
                tensor = F.batch_norm(tensor,
                                      layer.running_mean,
                                      layer.running_var,
                                      layer.weight,
                                      layer.bias,
                                      training=False,
                                      momentum=layer.momentum,
                                      eps=layer.eps)
            else:
                tensor = layer(tensor)
        return tensor

    def forward(self,
                entity_ebd,
                relation_ebd,
                query,
                predict_relation=False):
        """
        :param entity_ebd:Tensor, size=(num_entity, input_dim)
        :param relation_ebd: Tensor, size=(num_relation, input_dim)
        :param query: LongTensor, size=(num_query,2)
            - If predict_relation=False: query[:, 0] is head entity, query[:, 1] is relation
            - If predict_relation=True: query[:, 0] is head entity, query[:, 1] is tail entity
        :param predict_relation: bool, if True, predict relation instead of tail entity
        :return: Tensor,size=(num_query, num_entity) or (num_query, num_relation) depending on predict_relation
        """
        if predict_relation:
            # 预测关系：输入是 [头实体, 尾实体]
            x = torch.cat([entity_ebd.unsqueeze(dim=1)[query[:, 0]],
                           entity_ebd.unsqueeze(dim=1)[query[:, 1]]],
                          dim=1)
        else:
            # 预测尾实体：输入是 [头实体, 关系]
            x = torch.cat([entity_ebd.unsqueeze(dim=1)[query[:, 0]],
                           relation_ebd.unsqueeze(dim=1)[query[:, 1]]],
                          dim=1)
        x = self._apply_seq(self.block0, x)
        x.unsqueeze_(dim=1)
        x = self.block1(x)
        x = self._apply_seq(self.block2, x.squeeze(dim=2))
        norm_x = F.normalize(x, p=2, dim=1)
        if predict_relation:
            # 输出对所有关系的得分
            scores = torch.mm(x, relation_ebd.transpose(0, 1))
            cos_scores = torch.mm(norm_x, relation_ebd.transpose(0, 1))
        else:
            # 输出对所有实体的得分
            scores = torch.mm(x, entity_ebd.transpose(0, 1))
            cos_scores = torch.mm(norm_x, entity_ebd.transpose(0, 1))
        return scores,cos_scores


class WGCNEncoder(nn.Module):
    def __init__(self,
                 num_relation,
                 dim,
                 num_layer,
                 dropout=0.2):
        super(WGCNEncoder, self).__init__()

        self.num_layer = num_layer
        self.dropout_prop = dropout

        self.blocks = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.blocks.append(
                nn.Sequential(
                    nn.BatchNorm1d(dim),
                    nn.Tanh(),
                    nn.Dropout(self.dropout_prop)
                )
            )
            self.layers.append(WGCNLayer(num_relation, dim, dim))

    def forward(self,
                entity_embed,
                edge):
        h = entity_embed
        for i in range(self.num_layer):
            h = self.layers[i](h, edge)
            h = self.blocks[i](h)
        return h
