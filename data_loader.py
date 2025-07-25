from utils.data_process import load_data
from utils.data_process import load_dict
from utils.data_process import load_embedding
from utils.data_process import reverse_dict

from dateutil.relativedelta import relativedelta
import datetime as dt
from pathlib import Path
from collections import defaultdict
import csv
import os
import torch
class DataLoader(object):
    def __init__(self,
                 dataset,
                 dataset_file,
                 type,
                 event_name):
        """
        param dataset: the name of dataset
        param path: the path of dataset
        """
        self.type = type
        if dataset is None:
            raise Exception('You need to specify a dataset!')
        self.dataset = dataset
        self.dataset_file = dataset_file
        self.load_time = True
        if self.type.split('/')[0] == 'static':
            self.load_time = False

        # data
        self.device = 'cpu'
        self.data = []
        self.next_data = []
        self.prime_quadra = []
        self.num_entity = 0
        self.num_relation = 0
        self.num_time = 0
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.time2id = {}
        self.id2time = {}
        self.train = torch.LongTensor([])
        self.valid = torch.LongTensor([])
        self.test = torch.LongTensor([])
        self.predict = torch.LongTensor([])
        self.focus_en = {}
        self.en_embedding = []
        self.rel_embedding = []
        self.entities_cnt = defaultdict(int)
        self.Time =defaultdict(int)
        self.prev_result = {}
        self.now_result = {}
        self.predict_event = event_name
        self.predict_time = None

    def split_data(self,data, train_ratio=0.75, valid_ratio=0.125):
        train_size = int(len(data) * train_ratio)
        #valid_size = int(len(data) * valid_ratio)
        data = data.tolist()
        train_data = data[:train_size]
        # valid_data = data[train_size:train_size + valid_size]
        tmp = data[train_size:]
        valid_data = []
        for item in tmp:
            # if(self.entities_cnt[self.id2entity[item[0]]] <5 or self.entities_cnt[self.id2entity[item[2]]] < 5): 
            #     continue
            valid_data.append(item)

        # test_data = data[train_size + valid_size:]
        test_data = data[-1:]

        predict_data = []
        for idx in self.focus_en.keys() :
            idx = self.entity2id[idx]
            for i in range (self.num_relation):
                predict_data.append([idx, i, idx, 0])

        train_data = torch.LongTensor(train_data)
        valid_data = torch.LongTensor(valid_data)
        test_data = torch.LongTensor(test_data)
        predict_data = torch.LongTensor(predict_data)
        return train_data, valid_data,test_data,predict_data

    # 查找给定 key 的排序位置
    @staticmethod
    def find_key_position(sorted_items, key):
        for index, (k, v) in enumerate(sorted_items):
            if k == key:
                return index + 1  # 返回位置（从 1 开始计数）
        return 11111  # 如果 key 不存在，则返回 尽可能大的值
    
    def get_qradua(self,sorted_qradra = [],next_data = [],p_time = None):
        print("-------------获取四元组数据-------------")
        data = self.data = []
        tmp = self.num_relation
        
        # sorted_qradra,next_csv = self.read_csv(filename)
        for elements in sorted_qradra:
            self.entities_cnt[elements[0]] += 1
            self.entities_cnt[elements[3]] += 1
        
        # 更新最后预测时间
        if p_time is not None:
            self.predict_time = (p_time).strftime('%Y-%m-%d')
      
        # 存储后续事件的数据
        self.next_data = next_data
        print("-------------更新实体，关系和时间的数字映射-------------")
        for elements in sorted_qradra:
            if (len(elements) > 6): continue
            if (self.entity2id.get(elements[0]) == None):
                self.entity2id[elements[0]] = self.num_entity
                self.id2entity[self.num_entity] = elements[0]
                self.num_entity += 1
            if (self.entity2id.get(elements[3]) == None):
                self.entity2id[elements[3]] = self.num_entity
                self.id2entity[self.num_entity] = elements[3]
                self.num_entity += 1

            if (self.relation2id.get(elements[2]) == None):
                self.relation2id[elements[2]] = self.num_relation
                self.id2relation[self.num_relation] = elements[2]
                self.num_relation += 1

            if (self.time2id.get(elements[5]) == None):
                self.time2id[elements[5]] = self.num_time
                self.id2time[self.num_time] = elements[5]
                self.num_time += 1
            
            data.append([self.entity2id[elements[0]], self.relation2id[elements[2]],self.entity2id[elements[3]],self.time2id[elements[5]]])  
        
        if(self.num_relation != tmp):
            print("-------------更新预测集-------------")
            predict = []
            for idx in self.focus_en.keys() :
                idx = self.entity2id[idx]
                for i in range (self.num_relation):
                    predict.append([idx, i, idx, 0])
            self.predict = torch.LongTensor(predict)

        self.data = torch.LongTensor(data).to(self.device)
 
    def read_data(self,dataset,event_time = None,split_flag = False,mode = 'init'):
        sorted_quadra = []
        next_data = []
        
        # 对数据集去重排序
        # unique_dataset = list(set(map(tuple,dataset))) # 每次结果不一样
        unique_dataset = sorted(set(map(tuple,dataset)))
        sorted_dataset = sorted([list(elem) for elem in unique_dataset], key=lambda elem: elem[-1] )
        split_time = dt.datetime.strptime(sorted_dataset[-1][-1], "%Y-%m-%d").date()
        if self.predict_time is not None:
            pp_time = dt.datetime.strptime(self.predict_time, "%Y-%m-%d").date()
        #要划分数据
        if split_flag:
            #不知道事情发生的时间 拿1/3作为划分数据的依据
            length = len(sorted_dataset)//3
            start_time = dt.datetime.strptime(sorted_dataset[0][-1],"%Y-%m-%d").date()
            end_time = dt.datetime.strptime(sorted_dataset[-1][-1],"%Y-%m-%d").date()
            if event_time is None:
                # split_time = min(start_time + relativedelta(months=3) ,end_time)
                if end_time > start_time + relativedelta(months=5):
                    
                    split_time = start_time + relativedelta(months=3)
                else:
                    split_time = dt.datetime.strptime(sorted_dataset[length][-1], "%Y-%m-%d").date()
                pp_time = split_time
            #知道事情发生的时间  拿该时间作为划分数据的依据 
            else:
                if mode == 'init':
                    split_time = pp_time - relativedelta(months=3)
                else:
                    split_time = event_time
                
        
        for row in sorted_dataset:
            time_data = dt.datetime.strptime(row[5], "%Y-%m-%d").date()
            row[5] = time_data.strftime('%Y-%m-%d')
            # 将超过分界点的数据存储到next_data
            if(time_data>split_time): next_data.append(row)
            if(mode == 'fit' and  time_data > pp_time):
                row[5] = (time_data).strftime('%Y-%m-%d')
                sorted_quadra.append(row)
            if(mode == 'init' and time_data<=pp_time):
                row[5] = (time_data).strftime('%Y-%m-%d')
                sorted_quadra.append(row)
        
        return sorted_quadra,next_data,split_time
    
    def get_focus_en(self,label = None):
        if label is not None:
            self.focus_en = {}
            for item in label:
                # 将每个子列表按空格分隔成单词列表
                elements = item.split()
                # 确保至少有三个元素
                if len(elements) >= 3:
                    key = elements[0]  # 第一个元素作为键
                    value = elements[2]  # 第三个元素作为值
                    if key not in self.focus_en:
                        self.focus_en[key] = []  # 如果键还不存在，初始化为一个列表
                    if value not in self.focus_en[key]:  # 确保值不重复
                        self.focus_en[key].append(value)  # 将第三个元素加入对应键的值列表中
            print("-------------更新预测集-------------")
            predict = []
            for idx in self.focus_en.keys() :
                idx = self.entity2id[idx]
                for i in range (self.num_relation):
                    predict.append([idx, i, idx, 0])
            self.predict = torch.LongTensor(predict)

        else:
            KeyError("请提供决策标签，以获取关注实体")

    # 将新csv文件的数据更新到模型中
    def load_fit(self,sorted_qradra,next_csv,p_time):
        self.get_qradua(sorted_qradra,next_csv,p_time)
        self.train = self.valid
        self.valid = self.data

    def load(self,label,time = None):
        # 已知事件发生时间时，直接赋值
        e_time = None
        if time is not None:
            self.predict_time = time
            e_time = dt.datetime.strptime(time, "%Y-%m-%d").date()
        if len(self.dataset) < 10 : return 0
        print(f"--------处理 {self.predict_event} 数据--------")
        sorted_qradra, next_data, last_time = self.read_data(self.dataset,e_time,split_flag=True,mode = 'init')
        print("------------获取四元组数据------------")
        self.get_qradua(sorted_qradra,next_data,last_time)
        print("------------获取关注实体-------------")
        self.get_focus_en(label)
        #按照6:1:1划分数据
        print("-------------划分数据集-------------")
        self.train, self.valid,self.test,self.predict = self.split_data(self.data)
        print("-------------数据集处理完成-------------") 
        return 1

    def to(self,
           device):
        if device == 'cpu':
            self.train.cpu()
            self.valid.cpu()
            self.test.cpu()
            self.predict.cpu()
            torch.device('cpu')
        else:
            self.train = self.train.to(device)
            self.valid = self.valid.to(device)
            self.test = self.test.to(device)
            self.predict = self.predict.to(device)
            torch.device('cuda')
        self.device = device
