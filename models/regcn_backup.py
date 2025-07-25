import torch
import torch.nn as nn
from base_models.regcn_base import REGCNBase
from data_loader import DataLoader
from base_models.sacn_base import ConvTransEDecoder
import random
import utils.data_process as dps
from tqdm import tqdm
import utils.metrics as mtc
import numpy as np
import utils
from models.mate_model import MateModel
import json
import os
from utils.func import set_seed

Path = '/home/code/KGMH-main/lrd_test'
# Path = '/home/Projects/lrd_test'

class REGCN(MateModel):
    def __init__(self, model: REGCNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer,
                 ):
        super(REGCN, self).__init__()
        # common parameters
        self.model = model
        self.data = data
        self.opt = opt
        self.name = 'regcn'
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.predict_data = None
        self.train_time = None
        self.valid_time = None
        self.test_time = None
        self.pridict_time = None

        self.seq_len = model.seq_len    # time window
        self.grad_norm = 1.0

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.decoder = ConvTransEDecoder(model.hidden_dim,
                                         num_channel=50,
                                         kernel_length=3)
        self.opt.add_param_group({'params': self.decoder.parameters()})
        self.to(model.device)

    def data_process(self):
        # data process
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.predict_data, _, self.pridict_time = dps.split_data_by_time(self.data.predict)
    
    def update_embed(self):
        # 更新实体嵌入
        set_seed(0)
        if self.model.static_entity_embed.size(0) < self.data.num_entity:
            new_entity_embed = torch.nn.Parameter(torch.Tensor(self.data.num_entity, self.model.hidden_dim), requires_grad=True)
            # 使用 Xavier 初始化新嵌入
            torch.nn.init.xavier_uniform_(new_entity_embed)
            with torch.no_grad():
                new_entity_embed[:self.model.static_entity_embed.size(0)] = self.model.static_entity_embed
            self.model.static_entity_embed = new_entity_embed
            self.model.num_entity = self.data.num_entity
        # 更新关系嵌入
        if self.model.static_relation_embed.size(0) < self.data.num_relation * 2:
            new_relation_embed = torch.nn.Parameter(torch.Tensor(self.data.num_relation * 2, self.model.hidden_dim), requires_grad=True)
            # 使用 Xavier 初始化新嵌入
            torch.nn.init.xavier_uniform_(new_relation_embed)
            with torch.no_grad():
                new_relation_embed[:self.model.static_relation_embed.size(0)] = self.model.static_relation_embed
            self.model.static_relation_embed = new_relation_embed
            self.model.num_relation = self.data.num_relation 

    def train_epoch(self, batch_size=512,dataset='train'):
        self.data_process()
        self.train()
        self.opt.zero_grad()
        if dataset == 'train':
            t_data = self.train_data
        elif dataset == 'valid':
            t_data = self.valid_data
        # add reverse relation to graph
        data = dps.add_reverse_relation(t_data, self.data.num_relation)
        # target time for predict
        index = list(range(len(data)))
        random.shuffle(index)
        total_loss = 0

        for i in tqdm(index):
            if i == 0:
                # no history data
                continue
            # history data
            if i >= self.seq_len:
                edges = data[i - self.seq_len:i]
            else:
                edges = data[0:i]
            evolved_entity_embed, evolved_relation_embed = self.model.forward(edges)
            # put into a decoder to calculate score for object
            score,_ = self.decoder(evolved_entity_embed, evolved_relation_embed, data[i][:, :2])
            # calculate loss
            loss = self.loss(score, data[i][:, 2])
            loss.backward()
            total_loss = total_loss + float(loss)
            # clip gradient
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
            self.opt.step()
        
        return total_loss

    def test(self,
             batch_size=512,
             dataset='valid',
             tolerance = 0,
             filter_out=False,
             metric_list=None):
        if metric_list is None:
            metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mr', 'mrr']
        if dataset == 'valid':
            data = self.valid_data
            history = self.train_data
        elif dataset == 'test':
            data = self.test_data
            history = self.valid_data
        else:
            raise Exception
        # data = dps.add_reverse_relation(data, self.data.num_relation)
        if self.model.seq_len < len(history):
            history = dps.add_reverse_relation(history[-self.model.seq_len:],
                                               self.data.num_relation)
        else:
            history = dps.add_reverse_relation(history,
                                               self.data.num_relation)
        rank_list = []
        rank_list_filter = []
        self.eval()

        with torch.no_grad():
            evolved_entity_embed, evolved_relation_embed = self.model.forward(history)

            for edge in tqdm(data):
                score,_ = self.decoder(evolved_entity_embed, evolved_relation_embed, edge[:, [0,1]])
                # judge the rank of edge[:,2] among all entities, get top10 relative indicies 
                ranks,idx_score = mtc.calculate_rank(score, edge[:, 2], get_scores= True) 
                rank_list.append(ranks)     

                if filter_out:
                    ans = utils.data_process.get_answer(edge, self.data.num_entity, self.data.num_relation * 2)
                    score = utils.data_process.filter_score(score, ans, edge, self.data.num_relation * 2)
                    rank = mtc.calculate_rank(score, edge[:, 2])
                    rank_list_filter.append(rank)
    
        all_ranks = torch.cat(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_ranks)
        if filter_out:
            all_rank = np.concatenate(rank_list_filter)
            metrics_filter = mtc.ranks_to_metrics(metric_list, all_rank, filter_out)
            metrics.update(metrics_filter)
        return metrics

    def predict(self,
             output,
             batch_size=512,
             mode = 'prev',
             label = None,
             ):
        global Path
        self.data_process()
        data = self.predict_data
        
        if mode == 'prev':
            history = self.train_data
            p_score = self.data.prev_result
            time = self.data.id2time[int(self.data.train[-1][-1])]
        elif mode == 'fit' or mode == 'check':
            history = self.valid_data
            p_score = self.data.now_result
            time = self.data.predict_time
        else:
            raise Exception
        data = dps.add_reverse_relation(data, self.data.num_relation)
        if self.model.seq_len < len(history):
            history = dps.add_reverse_relation(history[-self.model.seq_len:],
                                               self.data.num_relation)
        else:
            history = dps.add_reverse_relation(history,
                                               self.data.num_relation)
        re_active = ["增进","感到满意","相信","认为优秀","欢迎","认为有成就","支持","认可","欣赏","视作英雄","喜欢","认为可靠","感谢","认为热情"]
        re_negative = ["担忧","损害","质疑","感到不满","认为非法","认为恐怖","威胁","攻击","认为缺乏","批评","认为有威胁","认为有危机","认为有暴力","认为犯罪","认为违规","认为失败"]
        top_10_entities= []
        self.eval()
        
        with torch.no_grad():
            evolved_entity_embed, evolved_relation_embed = self.model.forward(history)
            for edge in tqdm(data):
                score,cos_score = self.decoder(evolved_entity_embed, evolved_relation_embed, edge[:, [0,1]])
                # judge the rank of edge[:,2] among all entities, get top10 relative indicies 
                ranks,top_10_indices,top_10_s = mtc.calculate_rank(cos_score, edge[:, 2],get_top_10=True)  
                # rank_list.append(ranks)
            # new codes  
            # 将 top_10_indices 的每一行转换为对应的实体并插入到列表中
            for row in top_10_indices:
                entity_row = [self.data.id2entity[idx.item()] for idx in row]
                top_10_entities.append(entity_row)
        
            # for row in top_10_s:
            #     predict_en_score.append(row.sum(dim = 0).tolist())
            #     top_10_scores.append([idx.item() for idx in row])
            p_score.clear()
            # for k in self.data.focus_en.keys():
            #     p_score[k] = {}
            #     for en in self.data.focus_en[k]:
            #          # 记录每个实体对在当前时刻的默认得分
            #         p_score[k][en] = 0.0
            
            for i in range(0,top_10_s.size(0),self.data.num_relation):
                # 区分作为头实体还是尾实体，修改对应字典
                Fid = self.data.id2entity[edge[i][0].item()]
                p_score[Fid] = {}
                for en in self.data.focus_en[Fid]:
                    p_score[Fid][en] = 0.0
                for j in range (self.data.num_relation):
                    attitude = 0
                    if self.data.id2relation[j] in re_active:
                        attitude = 1 
                    elif self.data.id2relation[j] in re_negative:
                        attitude = -1
                            
                    for item,en in enumerate(top_10_entities[i+j]):
                        if en not in self.data.focus_en[Fid] or en == Fid : continue  # only focus on focus_en
                        num = (10-item) * top_10_s[i+j][item] * attitude
                        p_score[Fid][en] += float(num)
                        
            # with open(score_file,w_or_a) as s_f:
            #     s_f.write(f'{self.data.predict_event} {time} \n')
            #     for k in p_score.keys():
            #         for en in p_score[k].keys():
            #             s_f.write(f'{k} {en} {p_score[k][en]}\n')
        if mode == 'prev':
            time = self.data.id2time[int(self.data.train[-1][-1])]
            all_score = 0
            for k in p_score.keys():
                     for en in p_score[k].keys():
                        output[k+' -> '+en]={'x_data':[time],'y_data':[p_score[k][en]]}
                        all_score += p_score[k][en]

            output[self.data.predict_event] = {'x_data':[time],'y_data':[all_score]}
            output["predict"] = None

        # compare
        output_minmax = {} 
        if mode == 'fit' or mode == 'check':
            prev_score = self.data.prev_result
            now_score = self.data.now_result
            threshold = self.data.num_relation * 0.2

            all_diff = 0
            difference = 0
            if mode == 'check':
            # 当模式为检查时，默认预测结果为预测错误
                output["predict"] = False
            for k in now_score.keys():
                for en in now_score[k].keys():
                    difference = now_score[k][en] - prev_score[k][en]
                    if (k+" 不利好 "+en in label): sign = -1
                    else: sign = 1
                    all_diff += difference * sign

                    #记录该决策标签在对应时刻的值
                    output[k+' -> '+en]['x_data'].append(self.data.predict_time)
                    output[k+' -> '+en]['y_data'].append(float(difference))
                    
                    if mode == 'check':
                    # 所有标签的得分差值 只要有一个为正，即认为预测正确
                        if abs(difference) > threshold and  (difference > 0 and (k+" 利好 "+en in label) or difference < 0 and (k+" 不利好 "+en in label)):
                            output["predict"] = True
                    
            # 记录综合趋势在对应时刻的值
            output[self.data.predict_event]['x_data'].append(self.data.predict_time)
            output[self.data.predict_event]['y_data'].append(float(all_diff))
            #更新历史数据
            self.data.prev_result = now_score
            self.data.now_result = {}
        return output

    def loss(self, score, target):
        return self.cross_entropy_loss(score, target)

    def get_config(self):
        config = {}
        config['model'] = 'regcn'
        config['dataset'] = self.data.dataset_file
        config['num_entity'] = self.model.num_entity
        config['num_relation'] = self.model.num_relation
        config['hidden_dim'] = self.model.hidden_dim
        config['seq_len'] = self.model.seq_len
        config['num_layer'] = self.model.num_layer
        config['dropout'] = self.model.dropout_value
        config['active'] = self.model.if_active
        config['self_loop'] = self.model.if_self_loop
        config['layer_norm'] = self.model.layer_norm
        return config
