import os
import gc
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
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
import datetime as dt
from utils.func import set_seed

Path = '/home/code/KGMH-main/lrd_test'
# Path = '/home/Projects/lrd_test'
set_seed(0) 
MAXSIZE = 10000
class REGCN(MateModel):
    def __init__(self, model: REGCNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer,
                 ):
        super(REGCN, self).__init__()
        # common parameters
        self.model = model.double()
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

        self.cross_entropy_loss = nn.CrossEntropyLoss().double()
        self.decoder = ConvTransEDecoder(model.hidden_dim,
                                         num_channel=50,
                                         kernel_length=3).double()
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

    def train_epoch(self, batch_size=512,dataset='train',predict_flag = False):
        self.data_process()
        self.train()
        self.opt.zero_grad()
        if dataset == 'train':
            t_data = self.train_data
        elif dataset == 'valid':
            t_data = self.valid_data
        # 不添加反向关系
        data = t_data
        # target time for predict
        index = list(range(len(data)))
        # 打乱顺序
        # index = random.shuffle(index)
        # random.shuffle(index)
        total_loss = 0
        # print("index: ",index)

        for i in tqdm(index):
            if i == 0:
                # no history data
                continue
            # history data
            if i >= self.seq_len:
                edges = data[i - self.seq_len:i]
            else:
                edges = data[0:i]
            data_flag = False
            if data[i].size(0) > MAXSIZE:
                data_flag = True
            for j in range(0,data[i].size(0),MAXSIZE):
                evolved_entity_embed, evolved_relation_embed = self.model.forward(edges)
                
                end = min(j+MAXSIZE,data[i].size(0))
                # 使用头实体和尾实体预测关系
                # data[i][j:end, [0, 2]] 是 [头实体, 尾实体]
                # data[i][j:end, 1] 是关系（目标）
                score,_ = self.decoder(evolved_entity_embed, evolved_relation_embed, data[i][j:end, [0, 2]], predict_relation=True)
                loss = self.loss(score, data[i][j:end, 1])
                loss.backward()
                total_loss = total_loss + float(loss) 
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
                self.opt.step()
                if data_flag:
                    self.opt.zero_grad()
            
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
        if self.model.seq_len < len(history):
            history_edges = history[-self.model.seq_len:]
        else:
            history_edges = history
        rank_list = []
        rank_list_filter = []
        self.eval()

        with torch.no_grad():
            evolved_entity_embed, evolved_relation_embed = self.model.forward(history_edges)

            for edge in tqdm(data):
                score,_ = self.decoder(evolved_entity_embed,
                                       evolved_relation_embed,
                                       edge[:, [0, 2]],
                                       predict_relation=True)
                # judge the rank of edge[:,1] among all relations, get top10 relative indicies 
                ranks,idx_score = mtc.calculate_rank(score, edge[:, 1], get_scores= True) 
                rank_list.append(ranks)     

                if filter_out:
                    ans = utils.data_process.get_relation_answer(edge,
                                                                 self.data.num_entity,
                                                                 self.data.num_relation * 2)
                    score = utils.data_process.filter_relation_score(score,
                                                                     ans,
                                                                     edge,
                                                                     self.data.num_entity)
                    rank = mtc.calculate_rank(score, edge[:, 1])
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
             event_time = None
             ):
        global Path
        self.data_process()
        
        if mode == 'prev':
            history = self.train_data
            p_score = self.data.prev_result
            
        elif mode == 'fit' or mode == 'check':
            history = self.valid_data
            p_score = self.data.now_result
            time = self.data.predict_time
        else:
            raise Exception
        if self.model.seq_len < len(history):
            history_edges = history[-self.model.seq_len:]
        else:
            history_edges = history
        re_active = ["增进","感到满意","相信","认为优秀","欢迎","认为有成就","支持","认可","欣赏","视作英雄","喜欢","认为可靠","感谢","认为热情"]
        re_negative = ["担忧","损害","质疑","感到不满","认为非法","认为恐怖","威胁","攻击","认为缺乏","批评","认为有威胁","认为有危机","认为有暴力","认为犯罪","认为违规","认为失败"]
        self.eval()
        
        with torch.no_grad():
            evolved_entity_embed, evolved_relation_embed = self.model.forward(history_edges)
            device = evolved_entity_embed.device
            p_score.clear()
            query_pairs = []
            pair_entities = []
            for Fid in self.data.focus_en.keys():
                if Fid not in self.data.entity2id:
                    continue
                Fid_idx = self.data.entity2id[Fid]
                p_score[Fid] = {}
                for en in self.data.focus_en[Fid]:
                    p_score[Fid][en] = 0.0
                    if en not in self.data.entity2id or en == Fid:
                        continue
                    en_idx = self.data.entity2id[en]
                    query_pairs.append([Fid_idx, en_idx])
                    pair_entities.append((Fid, en))

            if query_pairs:
                query_tensor = torch.tensor(query_pairs, dtype=torch.long, device=device)
                _, cos_score = self.decoder(evolved_entity_embed,
                                              evolved_relation_embed,
                                              query_tensor,
                                              predict_relation=True)
                top_k = min(10, cos_score.size(1))
                top_values, top_indices = torch.topk(cos_score, k=top_k, dim=1)

                for idx, (Fid, en) in enumerate(pair_entities):
                    for rank_idx in range(top_k):
                        rel_idx = top_indices[idx, rank_idx].item()
                        base_idx = rel_idx % self.data.num_relation
                        relation_name = self.data.id2relation.get(base_idx)
                        attitude = 0
                        if relation_name in re_active:
                            attitude = 1
                        elif relation_name in re_negative:
                            attitude = -1
                        if attitude == 0:
                            continue
                        contribution = (10 - rank_idx) * top_values[idx, rank_idx].item() * attitude
                        # if contribution <0
                        #     print(f"贡献为负: {rank_idx} top_values[idx, rank_idx].item()} {relation_name} {contribution}")
                        p_score[Fid][en] += float(contribution)
        
            # with open(score_file,w_or_a) as s_f:
            #     s_f.write(f'{self.data.predict_event} {time} \n')
            #     for k in p_score.keys():
            #         for en in p_score[k].keys():
            #             s_f.write(f'{k} {en} {p_score[k][en]}\n')
        if mode == 'prev':
            if event_time is None:
                time = self.data.id2time[int(self.data.train[-1][-1])]
            else:
                time = event_time
            all_score = 0
            for k in p_score.keys():
                     for en in p_score[k].keys():
                        output[k+' -> '+en]={'x_data':[time],'y_data':[0]}
                        # all_score += p_score[k][en]

            output[self.data.predict_event] = {'x_data':[time],'y_data':[0]}
            output["predict"] = None

        # compare
        output_minmax = {} 
        if mode == 'fit' or mode == 'check':
            prev_score = self.data.prev_result
            now_score = self.data.now_result
            threshold = self.data.num_relation * 0.2

            all_diff = 0
            difference = 0
            if event_time is None:
                event_time = self.data.predict_time
            if mode == 'check':
            # 当模式为检查时，默认预测结果为预测错误
                output["predict"] = False
            for k in now_score.keys():
                for en in now_score[k].keys():
                    # difference = now_score[k][en] - prev_score[k][en] // 差值画图
                    difference = now_score[k][en]
                    # if (k+" 不利好 "+en in label): sign = -1
                    # else: sign = 1
                    # all_diff += difference * sign  // 应该不需要这个符号，正值表示利好，负值表示不利好
                    all_diff += difference

                    #记录该决策标签在对应时刻的值
                    
                    output[k+' -> '+en]['x_data'].append(event_time)
                    output[k+' -> '+en]['y_data'].append(float(difference))
                    
                    if mode == 'check':
                    # 所有标签的得分差值 只要有一个为正，即认为预测正确
                        if abs(difference) > threshold and  (difference > 0 and (k+" 利好 "+en in label) or difference < 0 and (k+" 不利好 "+en in label)):
                            output["predict"] = True
                    
            # 记录综合趋势在对应时刻的值
            output[self.data.predict_event]['x_data'].append(event_time)
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
