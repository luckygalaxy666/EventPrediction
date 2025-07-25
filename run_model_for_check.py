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
# mongo_host = "121.48.163.69"
# mongo_port = 27018
# db_name = "TSKGSystem"
# collection_name = "eventPredictionEv"
# username = "root"
# password = "rootdb@"
# auth_source = "admin"  # 使用的身份验证数据库

# 连接到MongoDB
# MONGO_Ti_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionTi']
# Path to the main.py file
PATH = '/root/lrd_test'
dir_path = 'tsmc_es_data'
label_path = "tsmc_label.json"
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
        if event_time in graph_data['x_data']:
            ax.axvline(x=event_time, color='red', linestyle='--', label=f"事件: {event_time}")
         # 设置x轴标签，避免重叠
        x_data = graph_data['x_data']
        # 控制 x 轴显示的标签数量，例如每隔 2 个日期显示一个标签
        step = max(1, len(x_data) // 10)  # 根据数据量动态选择显示的间隔
        ax.set_xticks(x_data[::step])  # 只显示部分日期

        # 自动旋转 x 轴标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")



    # 调整布局，避免标签重叠
    plt.tight_layout()
    with open(file+"/output.png", 'wb') as f:
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


def run_command(dataset_file,event_name,mode,timespan,label,event_time,checkpoint,continue_flag = False, process_chinese_processing = False):
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
    if process_chinese_processing:
        command+=["--process_chinese_processing"]
    print(command)
    
    print(f"Running command for {event_name}...")
    subprocess.run(command)
    print(f"Finished running for {event_name}\n")
def main(event_name=None, mode='check', timespan='31', process_chinese_processing=False, csv_file_path=None):
    """
    主函数
    
    Args:
        event_name (str): 事件名称，如果为None则使用默认事件
        mode (str): 运行模式
        timespan (str): 时间跨度
    """
    import argparse
    
    # 如果没有提供参数，则解析命令行参数
    if event_name is None:
        parser = argparse.ArgumentParser(description='运行模型检查')
        parser.add_argument('--event_name', help='事件名称')
        parser.add_argument('--mode', default='check', choices=['check', 'fit', 'init'], help='运行模式')
        parser.add_argument('--timespan', default='31', help='时间跨度')
        args = parser.parse_args()
        
        event_name = args.event_name
        mode = args.mode
        timespan = args.timespan
    
    # 如果没有指定事件名称，使用默认事件
    if event_name is None:
        event_name = '美国对台军售'
    
    # 加载JSON 数据
    event_label = load_json(label_path)
    failed = []
    folder_path = "./"+ dir_path
    
    # 构建文件名
    file = f'{event_name}.csv'
    
    try:
        event = Path(file).stem
        try:
            label = event_label[event]["tags"]
            event_time = event_label[event]["time"]
        except KeyError:
            print(f"警告: 事件 '{event}' 在标签文件中未找到，使用默认标签")
            label = ["美国 利好 台湾", "台湾 利好 美国"]
            event_time = 'Unknown'
            failed.append(event)
        
        # 使用传入的CSV文件路径，如果没有传入则使用默认路径
        if csv_file_path is not None and os.path.exists(csv_file_path):
            dataset_file = csv_file_path
            print(f"使用指定的CSV文件: {dataset_file}")
        else:
            dataset_file = os.path.join(folder_path, file)
            print(f"使用默认CSV文件: {dataset_file}")
        
        # 检查文件是否存在
        if not os.path.exists(dataset_file):
            print(f"错误: 数据文件不存在: {dataset_file}")
            return
        
        # 将时间转换为标准datetime格式
        dataset = read_csv(dataset_file)
        # 如果使用的是处理后的文件，不需要重新保存
        if csv_file_path is None or csv_file_path == os.path.join(folder_path, file):
            load_csv(dataset, event_name)
        
        if mode == 'check':
            checkpoint = 'base_model'
        elif mode == 'fit':
            if event_name in event_label:
                checkpoint = 'base_model'
            else:
                checkpoint = 'latest_model'
        else: 
            checkpoint = "None"
        
        run_command(dataset_file, event_name, mode, timespan, label, event_time, checkpoint, process_chinese_processing=process_chinese_processing)
        
        if mode == 'init': # 取一部分数据训练好基模型后，继续取剩下的数据拟合模型，输出结果
            run_command(dataset_file, event_name, 'fit', timespan, label, event_time, 'base_model', continue_flag=True, process_chinese_processing=process_chinese_processing)
        
        # 计算输出目录路径
        if csv_file_path is not None and csv_file_path != os.path.join(folder_path, file):
            # 如果使用了处理后的文件，输出目录基于事件名称
            output_dir = os.path.join(folder_path, event_name)
        else:
            # 使用默认路径
            output_dir = dataset_file[:-4]
        
        output = load_json(output_dir + "/output.json")
        # if mode == 'check':
        draw_output(output, event_time, output_dir)
            
    except Exception as e:
        print(f"处理事件 '{event_name}' 时出错: {e}")
        failed.append(event_name)
    
    if failed:
        print(f"失败的事件: {failed}")

                
if __name__ == "__main__":
    main()