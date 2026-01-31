import csv
import pymongo
import os
import json
from datetime import datetime
# # MongoDB配置
# mongo_host = "121.48.163.69"
# mongo_port = 27018
# db_name = "TSKGSystem"
# collection_name = "eventPredictionEv"
# username = "root"
# password = "rootdb@"
# auth_source = "admin"  # 使用的身份验证数据库

# # 连接到MongoDB
# MONGO_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionEv']
# MONGO_Ti_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionTi']

BATCH_SIZE = 1000
with open("new_label.json", encoding='utf-8') as f:
    data = json.load(f)

# 用于存储要插入的数据的列表
documents = []
events = []
# MONGO_Ti_DB.delete_many({})
# 解析数据
for event_name, details in data.items():
    time = details["time"]
    events.append(event_name)
    for tag in details["tags"]:
        source_label, edge_label, target_label = tag.split()  # 假设标签以空格分隔
        document = {
            "source_label": source_label,
            "target_label": target_label,
            "edge_label": edge_label,
            "event_name": event_name,
            "modify_time": time,
            "model_type":"check"
        }
        documents.append(document)

# 批量插入数据到MongoDB
# if documents:
#     MONGO_Ti_DB.insert_many(documents)

# print(f"{len(documents)} 条数据已成功插入到MongoDB!")
name =[]
label = []
# MONGO_DB.delete_many({})
for root, dirs, files in os.walk("./new_data"):
    for file in files:
        if file.endswith(".csv"):
# # CSV文件路径
# csv_file_path = "./temporal_data/美国派前国防官员访台.csv"  # 替换为你的CSV文件路径

# 获取event_name为CSV文件的前缀
            event_name = os.path.splitext(os.path.basename(file))[0]
            if event_name not in events:
                # os.remove("./temporal_data/"+file)
                # print(f"Deleted event: {event_name}")
                continue
            name.append(event_name)
            # print(event_name)
            date_obj = datetime.strptime(data[event_name]["time"], "%Y-%m-%d")
            label.append(date_obj.strftime("%Y年%-m月%-d日"))
            # print(date_obj.strftime("%Y年%-m月%-d日"))
            print(event_name)
            # MONGO_DB.delete_many({})
            # 读取CSV文件并插入到MongoDB

           