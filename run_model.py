import os

import subprocess
import json
import csv 
from pathlib import Path
import pymongo
import datetime as dt
# # MongoDB配置
# mongo_host = "121.48.163.69"
# mongo_port = 27018
# db_name = "TSKGSystem"
# collection_name = "eventPredictionEv"
# username = "root"
# password = "rootdb@"
# auth_source = "admin"  # 使用的身份验证数据库

# # 连接到MongoDB
# MONGO_Ti_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionTi']
# # Path to the main.py file
PATH = '/home/code/KGMH-main/lrd_test'
Path = '/home/Projects/lrd_test'
dir_path = 'temporal_data'
# 加载并读取JSON数据的函数
def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def get_dataset(link,node):
    dataset=[]
    # 创建id到名称和类型的映射
    id_to_name_type = {n["id"]: (n["name"], n["type"]) for n in node}
    # 转换为所需的dataset格式
    dataset = []
    for l in link:
        source_id = l["source_id"]
        target_id = l["target_id"]
        
        source_name, source_type = id_to_name_type.get(source_id, ("UNK", "UNK"))
        target_name, target_type = id_to_name_type.get(target_id, ("UNK", "UNK"))
        
        # 将每条记录转化为指定的格式 ["source_name", "Event", "type", "target_name", "target_type", "timeDate"]
        time= dt.datetime.strptime(l["timeDate"], "%Y-%m-%d").date().strftime('%Y-%m-%d')
        dataset.append([source_name,source_type, l["type"], target_name, target_type, time])
    return dataset
def read_csv(path_csv):
    data = []
    with open(path_csv,"r",encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            data.append(row)
    return data

def get_en_cnt(dataset,mode,file):
    
    entitys_cnt = {}
    if mode == 'fit':
        with open(file, 'r',encoding='utf-8') as f:
            entitys_cnt = json.load(f)
    for row in dataset:
        source_name = row[0]
        target_name = row[3]
        if source_name not in entitys_cnt:
            entitys_cnt[source_name] = 0
        if target_name not in entitys_cnt:
            entitys_cnt[target_name] = 0
        entitys_cnt[source_name] += 1
        entitys_cnt[target_name] += 1
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w',encoding='utf-8') as f:
        json.dump(entitys_cnt, f)
    return entitys_cnt

def load_csv(dataset,event_name,mode):
    dataset_file = "./"+dir_path+"/"+event_name+".csv" 
    entitys_cnt_file = "./"+dir_path+"/"+event_name+"/entitys_cnt.json"
    if mode == 'check': return dataset_file
    else:
        entitys_cnt = get_en_cnt(dataset,mode,entitys_cnt_file)


    with open(dataset_file, 'w',encoding='utf-8') as f:
        for row in dataset:
            if entitys_cnt[row[0]] < 3 or entitys_cnt[row[3]] < 3:
                continue
            time=dt.datetime.strptime(row[5], "%Y-%m-%d").date()
            row[5] = time.strftime('%Y-%m-%d')
            f.write(','.join(row) + '\n')
    return dataset_file


def run_command(dataset_file,event_name,mode,timespan,label,event_time,checkpoint,continue_flag = False):
    if mode == 'init':
        batch_size = 256
    else:
        batch_size = 1024
    command = [
        "python", PATH +"/main.py",
        "--dataset", dataset_file,
        "--model", "regcn",
        "--weight-decay", "1e-5",
        "--early-stop", "3",
        "--train",
        "--mode",mode,
        "--label", str(label),
        "--timespan",str(timespan),
        "--event_name",event_name,
        "--dir_path",dir_path,
        "--batch-size",str(batch_size)
    ]
    if continue_flag is True:
        command+=["--continue_flag"]
    if event_time != ' 'and  event_time != '':
        command+=["--time",event_time]
    if mode == 'fit' or mode == 'check':
        command+=["--checkpoint",checkpoint]
    
    
    print(f"Running command for {event_name}...")
    subprocess.run(command)
    print(f"Finished running for {event_name}\n")
def main(input_json):

    # 加载JSON 数据
    data = load_json(input_json)
    event_label = load_json(PATH + "/label.json")
    new_label = load_json(PATH + "/new_label.json")
    try:
        link = data['link']
        node = data['node']
        mode = data['mode']
        timespan = data['timespan']
        event_name = data['model_name']
        label = [f"{item['source']} {'1'} {item['target']}" for item in data['label']]
        event_time = data['event_time']
    except:
        raise KeyError("输入条件有误！")
    
    entitys = [n["name"] for n in node]
    documents = []  # 统计某事件 init模式下的标签 传入数据库
    for item in data['label']:
        if item['source'] not in entitys or item['target'] not in entitys:
            raise KeyError("标签中的实体没有出现在子图中，请重新输入标签！")
        else:
            document = {
            "source_label": item['source'],
            "target_label": item['target'],
            "edge_label": ' ',
            "event_name": event_name,
            "modify_time": '',
            "model_type":"init"
        }
            documents.append(document)
    dataset = get_dataset(link,node)
    
    global dir_path
    
    if event_name in event_label:
        dir_path = 'temporal_data'
        label = event_label[event_name]["tags"]
     
    if event_name in new_label:
        dir_path = 'new_data'
        label = new_label[event_name]["tags"]   

    dataset_file = load_csv(dataset,event_name,mode)
    if mode == 'check':
        checkpoint = 'base_model'
    elif mode  == 'fit':
        if event_name in event_label:
            checkpoint = 'base_model'
        else:
            checkpoint = 'latest_model'
    else: 
        checkpoint = "None"
        # MONGO_Ti_DB.insert_many(documents)
        # print(f"{event_name}的{len(documents)} 个标签已成功插入到eventPredictionEv!")
   
    run_command(dataset_file,event_name,mode,timespan,label,event_time,checkpoint)
    if mode == 'init': # 取一部分数据训练好基模型后，继续取剩下的数据拟合模型，输出结果
        run_command(dataset_file,event_name,'fit',timespan,label,event_time,'base_model',continue_flag = True)
    file =dataset_file[:-4]
    output = load_json(file+ "/output.json")
    return output
                
if __name__ == "__main__":
    input = PATH + "/init_input.json"
    output = main(input)