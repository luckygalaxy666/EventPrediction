import os
import subprocess
import json
import csv 
from pathlib import Path
import pymongo
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/SimHei.ttf'
fm.fontManager.addfont(font_path)

from matplotlib import rcParams

# 设置字体为支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False    # 正常显示负号
# MongoDB配置
mongo_host = "121.48.163.69"
mongo_port = 27018
db_name = "TSKGSystem"
collection_name = "eventPredictionEv"
username = "root"
password = "rootdb@"
auth_source = "admin"  # 使用的身份验证数据库

# 连接到MongoDB
MONGO_Ti_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionTi']
# Path to the main.py file
PATH = '/home/code/KGMH-main/lrd_test'
dir_path = 'new_data'
label_path = "new_label.json"
# Path = '/home/Projects/lrd_test'
# 加载并读取JSON数据的函数
def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def draw_output(data,event_time,file):

    # 设置图的布局
    fig, axs = plt.subplots(len(data) - 1, 1, figsize=(10, 6 * (len(data) - 1)))

    # 处理每个图表
    for i, graph_data in enumerate(data[:-1]):  # 排除最后一个预测项
        ax = axs[i]  # 获取对应的子图
        ax.plot(graph_data['x_data'], graph_data['y_data'], marker='o')
        ax.set_title(graph_data['lineGraphName'])
        ax.set_xlabel('时间')
        ax.set_ylabel('概率')
        ax.grid(True)
        ax.axvline(x="2024-09-13", color='red', linestyle='--', label=f"事件: {event_time}")
        # if event_time in graph_data['x_data']:
        #     ax.axvline(x=event_time, color='red', linestyle='--', label=f"事件: {event_time}")
         # 设置x轴标签，避免重叠
        x_data = graph_data['x_data']
        # 控制 x 轴显示的标签数量，例如每隔 2 个日期显示一个标签
        step = max(1, len(x_data) // 10)  # 根据数据量动态选择显示的间隔
        ax.set_xticks(x_data[::step])  # 只显示部分日期

        # 自动旋转 x 轴标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")



    # 调整布局，避免标签重叠
    plt.tight_layout()
    with open(file+"/output_diy.png", 'wb') as f:
        plt.savefig(f)
    print(f"图表已保存为: {file}/output.png")
    plt.close()

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
        dataset.append([source_name,source_type, l["type"], target_name, target_type, l["timeDate"]])
    return dataset
def read_csv(path_csv):
    data = []
    with open(path_csv,"r",encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if len(row)!=6:
                continue
            data.append(row)
    print(len(data))
    return data

def load_csv(dataset,event_name):
    dataset_file = "./"+dir_path + "/"+event_name+".csv" 
    # if os.path.exists(dataset_file): return dataset_file
    with open(dataset_file, 'w',encoding='utf-8') as f:
        for row in dataset:
            time=dt.datetime.strptime(row[5], "%Y-%m-%d").date()
            row[5] = time.strftime('%Y-%m-%d')
            f.write(','.join(row) + '\n')
    return dataset_file


def run_command(dataset_file,event_name,mode,timespan,label,event_time,checkpoint,continue_flag = False):
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
        "--dir_path",dir_path
    ]
    if continue_flag is True:
        command+=["--continue_flag"]
    if event_time != 'Unknown':
        command+=["--time",event_time]
    if mode == 'fit' or mode == 'check':
        command+=["--checkpoint",checkpoint]
    
    
    print(f"Running command for {event_name}...")
    subprocess.run(command)
    print(f"Finished running for {event_name}\n")
def main(input_json):

    # 加载JSON 数据
    event_label = load_json(label_path)
    # 遍历文件夹，找到所有的 .csv 文件
    failed = []
    folder_path = "./"+ dir_path
    mode = 'check'
    timespan = '31'
    # focus_en = ["美国", "中国", "赖清德", "蔡英文", "台湾", "拜登"]
    # label = []
    # for f_en in focus_en:
    #     for s_en in focus_en:
    #         if f_en != s_en:
    #             label.append(f"{f_en} 利好 {s_en}")

    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         if file.endswith(".csv"):
    file = '赖清德出席典礼.csv'
    if(True):
                event= Path(file).stem
                # event = prefix.split('.')[0]
                try:
                    label = event_label[event]["tags"]
                    event_time  = event_label[event]["time"]
                except:
                    failed.append(event)
                    # continue
                dataset_file = os.path.join(folder_path, file)

                # os.makedirs(dataset_file, exist_ok=True)
                event_name = event
                # 将时间转换为标准datetime格式
                dataset = read_csv(dataset_file)
                load_csv(dataset,event_name)
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
                # label =["赖清德 不利好 中国"]
                # event_time = 'Unknown'
                # run_command(dataset_file,event_name,mode,timespan,label,event_time,checkpoint)
                # if mode == 'init': # 取一部分数据训练好基模型后，继续取剩下的数据拟合模型，输出结果
                #     run_command(dataset_file,event_name,'fit',timespan,label,event_time,'base_model',continue_flag = True)
                file =dataset_file[:-4]
                # output = []
                output = load_json(file+ "/output.json")
                if mode == 'check':
                    draw_output(output,event_time,file)

                
if __name__ == "__main__":
    input = PATH + "/11111.json"
    main(input)