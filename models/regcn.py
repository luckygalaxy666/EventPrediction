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
import networkx as nx
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

# 【最终修正版】基于列表索引的时序加权热度计算
    def calculate_dynamic_heat(self, time_slice_edges, decay_factor=0.1):
        """
        计算特定时间切片内的实体热度。
        利用列表索引作为时间步，越靠后的元素代表越近的时间，权重越大。
        """
        import torch
        import numpy as np
        import networkx as nx
        from collections import defaultdict

        try:
            # 初始化带权边字典: {(src, dst): accumulated_weight}
            weighted_edges = defaultdict(float)
            
            # 1. 如果是列表格式 (List[Tensor])
            if isinstance(time_slice_edges, list):
                num_slices = len(time_slice_edges)
                
                for i, tensor in enumerate(time_slice_edges):
                    # 判空与类型安全检查
                    if tensor is None or not hasattr(tensor, 'cpu'): continue
                    
                    edges_np = tensor.detach().cpu().numpy()
                    if edges_np.size == 0: continue
                    
                    # 核心逻辑：计算时序权重
                    # i=0 (最旧) -> dt = max -> weight = min
                    # i=last (最近) -> dt = 0 -> weight = 1.0
                    dt = num_slices - 1 - i
                    time_weight = np.exp(-decay_factor * dt)
                    
                    # 提取源和目的节点
                    # 假设 Tensor 格式为 [s, r, o] 或 [s, r, o, t]
                    src_nodes = edges_np[:, 0]
                    dst_nodes = edges_np[:, 2]
                    
                    # 累加权重 (处理同一边在不同时间步出现的情况)
                    for s, d in zip(src_nodes, dst_nodes):
                        weighted_edges[(s, d)] += time_weight
                        
            # 2. 如果是单个 Tensor (无时序信息，退化为普通图)
            elif hasattr(time_slice_edges, 'cpu'):
                edges_np = time_slice_edges.detach().cpu().numpy()
                src_nodes = edges_np[:, 0]
                dst_nodes = edges_np[:, 2]
                for s, d in zip(src_nodes, dst_nodes):
                    weighted_edges[(s, d)] += 1.0
            
            else:
                return {}

            # 3. 构建 NetworkX 图
            G = nx.DiGraph()
            if not weighted_edges:
                return {}
                
            # 批量添加带权边
            edge_list = [(u, v, w) for (u, v), w in weighted_edges.items()]
            G.add_weighted_edges_from(edge_list)

            # 4. 计算带权 PageRank
            try:
                # weight='weight' 指定使用我们累加的时序权重
                heat_scores = nx.pagerank(G, alpha=0.85, weight='weight')
            except:
                heat_scores = nx.degree_centrality(G)

            return heat_scores

        except Exception as e:
            print(f"Error in calculate_dynamic_heat: {e}")
            return {}

    def predict(self, output, batch_size=512, mode='prev', label=None, event_time=None):
        self.data_process()
        
        # 1. 确定当前预测所依据的历史数据（这就是你的“时间切片”）
        if mode == 'prev':
            history = self.train_data
            p_score = self.data.prev_result
        elif mode == 'fit' or mode == 'check':
            history = self.valid_data
            p_score = self.data.now_result
        else:
            raise Exception("Unknown mode")

        # 2. 截取滑动窗口作为“当前时间片”
        # self.model.seq_len 代表模型回溯的历史长度（例如过去3天或过去3个快照）
        # 这段 history_edges 就是模型“眼中”的当前局势
        if self.model.seq_len < len(history):
            current_time_slice = history[-self.model.seq_len:]
        else:
            current_time_slice = history
        
        # =======================================================
        # 核心步骤：计算当前时间片下的动态热度
        # 这会导致不同时间点（随着 predict 被多次调用），热度值不同
        # =======================================================
        current_heat_map = self.calculate_dynamic_heat(current_time_slice, decay_factor=0.1)

        # 定义积极/消极关系词表（用于双边关系态势度量）
        re_active = ["增进", "感到满意", "相信", "认为优秀", "欢迎", "支持", "认可", "欣赏", "感谢"]
        re_negative = ["担忧", "损害", "质疑", "感到不满", "认为非法", "威胁", "攻击", "批评", "制裁"]

        self.eval()
        with torch.no_grad():
            # 模型前向传播
            evolved_entity_embed, evolved_relation_embed = self.model.forward(current_time_slice)
            device = evolved_entity_embed.device
            p_score.clear()
            
            # --- 优化点：预计算关系态势向量 ---
            # 创建一个 [num_relations] 的向量，存储每个关系的态势分 (+1/-1/0)
            # 这样查表比 if string in list 快得多
            num_rels = len(self.data.id2relation)
            stance_vector = torch.zeros(num_rels, device=device)
            
            for rel_id, rel_name in self.data.id2relation.items():
                if rel_name in re_active:
                    stance_vector[rel_id] = 1.0
                elif rel_name in re_negative:
                    stance_vector[rel_id] = -1.0
            
            # 准备查询对
            query_pairs = []
            pair_entities = [] # 记录 (Fid, en) 原始名称
            
            for Fid in self.data.focus_en.keys():
                if Fid not in self.data.entity2id: continue
                Fid_idx = self.data.entity2id[Fid]
                p_score[Fid] = {}
                
                for en in self.data.focus_en[Fid]:
                    p_score[Fid][en] = 0.0 # 初始化
                    if en not in self.data.entity2id or en == Fid: continue
                    en_idx = self.data.entity2id[en]
                    
                    query_pairs.append([Fid_idx, en_idx])
                    pair_entities.append((Fid, en))

            if query_pairs:
                query_tensor = torch.tensor(query_pairs, dtype=torch.long, device=device)
                
                # Decoder 计算
                _, cos_score = self.decoder(evolved_entity_embed, 
                                          evolved_relation_embed, 
                                          query_tensor, 
                                          predict_relation=True)
                
                # 获取 Top-K 关系索引
                top_k = min(10, cos_score.size(1))
                # top_indices shape: [num_pairs, k]
                top_values, top_indices = torch.topk(cos_score, k=top_k, dim=1)

                # --- 优化点：向量化计算分数 ---
                
                # 1. 查表获取 Top-K 关系的态势分
                # shape: [num_pairs, k]
                batch_stance_scores = stance_vector[top_indices]
                
                # 2. 对 Top-K 求和，得到每个 pair 的基础态势分
                # shape: [num_pairs]
                base_scores = batch_stance_scores.sum(dim=1)
                
                # 3. 准备热度系数
                # 【核心修改】准备双向热度系数
                heat_values = []
                for sub_name, obj_name in pair_entities:
                    # 获取 ID
                    sub_id = self.data.entity2id[sub_name]
                    obj_id = self.data.entity2id[obj_name]
                    
                    # 查表获取热度 (如果不在当前子图中，热度为0)
                    h_sub = current_heat_map.get(sub_id, 0.0)
                    h_obj = current_heat_map.get(obj_id, 0.0)
                    
                    # --- 融合策略 ---
                    # 策略：累加主客体热度。
                    # 逻辑：只要有一方是热点，这个交互就值得被放大；如果双方都是热点，则极度放大。
                    combined_heat = h_sub + h_obj
                    heat_values.append(combined_heat)
                
                heat_tensor = torch.tensor(heat_values, device=device)
                
                # 4. 计算最终融合分数: Score = Base * (1 + 10 * Heat)
                # 使用 tensor 乘法一次性完成
                fusion_factors = 1.0 + (heat_tensor * 10.0)
                final_scores = base_scores * fusion_factors
                
                # 5. 写回结果字典 (这一步必须串行，但数据量已经很小了)
                final_scores_cpu = final_scores.cpu().numpy()
                for i, (Fid, en) in enumerate(pair_entities):
                    p_score[Fid][en] = float(final_scores_cpu[i])
                    
                    
                #     p_score[Fid][en] += sentiment_score
                print("Fid -> en: ", Fid, "->", en)
                print("sentiment_score: ", p_score[Fid][en])
                print("heat_weight: ", heat_tensor)
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
            # difference = 0
            if event_time is None:
                event_time = self.data.predict_time
            if mode == 'check':
            # 当模式为检查时，默认预测结果为预测错误
                output["predict"] = False
            for k in now_score.keys():
                for en in now_score[k].keys():
                    # all_diff += abs(now_score[k][en])   
                    # difference = now_score[k][en] - prev_score[k][en] // 差值画图
                    # difference = now_score[k][en]
                    if (k+" 不利好 "+en in label): sign = -1
                    else: sign = 1
                    all_diff += now_score[k][en] * sign  
                    # all_diff += difference

                    #记录该决策标签在对应时刻的值
                    
                    output[k+' -> '+en]['x_data'].append(event_time)
                    output[k+' -> '+en]['y_data'].append(float(now_score[k][en]))
                    
                    if mode == 'check':
                    # 所有标签的得分差值 只要有一个为正，即认为预测正确
                        if abs(now_score[k][en]) > threshold and  (now_score[k][en] > 0 and (k+" 利好 "+en in label) or now_score[k][en] < 0 and (k+" 不利好 "+en in label)):
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
