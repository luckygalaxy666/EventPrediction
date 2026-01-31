import csv
import pymongo
import os
import json

keywords = {
    "美国对台军售": ["军售"],
    "赖清德纪念金门战役": ["金门","古宁头"],
    "台方支持美军舰入境": ["舰"],
    "联合利剑演习": ["演习","军队","利剑"],
    "台洪断交": ["断交","洪","宏都拉斯"],
    "台舰新建计划": ["舰","海巡"],
    "赖清德视察演习军队": ["演习","军队","利剑"],
    "赖清德与无国界记者组织会晤": ["记者","无国界"],
    "赖清德接见瓜地马拉外宾": ["瓜地马拉","外宾"],
    "蔡英文国庆发言": ["国庆","蔡英文"],
    "双十讲话": ["双十","新两国论"],
    "赖清德接见露西亚参议长": ["露西亚","参议长","雷诺丝","法兰西斯"],
    "赖清德支持访欧": ["欧洲","访欧"],
    "赖清德维护主权": ["主权","国庆"],
    "美国军事援助": ["军事","援助","军援","军售"],
    "全社会防卫韧性": ["防卫","韧性","国防"],
    "台方感谢军售": ["军售","感谢"],
    "赖清德出席典礼":[ "典礼","新生","学校"],
    "赖清德接见吐瓦鲁国会议长": ["吐瓦鲁","议长","伊塔雷理","国会"],
    "赖清德视察澎湖军队": ["澎湖","军队","部队"],
    "台政府参与联合国案": ["联合国","政府","古特雷斯"],
}

common_keywords = ["赖清德","蔡英文","美国","中国","拜登","台湾"]
# MongoDB配置
mongo_host = "121.48.163.69"
mongo_port = 27018
db_name = "TSKGSystem"
collection_name = "eventPredictionEv"
username = "root"
password = "rootdb@"
auth_source = "admin"  # 使用的身份验证数据库

# 连接到MongoDB
MONGO_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionEv']
MONGO_Ti_DB = pymongo.MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['TSKGSystem']['eventPredictionTi']

BATCH_SIZE = 1000
with open("new_label.json", encoding='utf-8') as f:
    new_data = json.load(f)

with open("label.json", encoding='utf-8') as f:
    data = json.load(f)

data.update(new_data)

for doc in MONGO_Ti_DB.find():
    event_name = doc.get('event_name', None)  # 默认为 None，如果没有该字段
    # print("!!!",doc)
    # print(event_name)
    # 如果没有 event_name 字段，跳过该文档
    if event_name not in data.keys():
        # MONGO_Ti_DB.delete_one({"_id": doc["_id"]})
        print(doc)
        print(f"删除了event_name为 {event_name} 的文档")

# 用于存储要插入的数据的列表
# documents = []
# events = []
# MONGO_Ti_DB.delete_many({})
# # 解析数据
# for event_name, details in data.items():
#     time = details["time"]
#     events.append(event_name)
#     # for tag in details["tags"]:
#     #     source_label, edge_label, target_label = tag.split()  # 假设标签以空格分隔
#     #     document = {
#     #         "source_label": source_label,
#     #         "target_label": target_label,
#     #         "edge_label": edge_label,
#     #         "event_name": event_name,
#     #         "modify_time": time,
#     #         "model_type":"check"
#     #     }
#     #     documents.append(document)
#     for keyword in keywords[event_name]:
#         for common_keyword in common_keywords:
#             document = {
#                 "source_label": common_keyword,
#                 "target_label": keyword,
#                 "edge_label": "相关",
#                 "event_name": event_name,
#                 "modify_time": time,
#                 "model_type":"check"
#             }
#             documents.append(document)
            
        

# # 批量插入数据到MongoDB
# if documents:
#     MONGO_Ti_DB.insert_many(documents)

# print(f"{len(documents)} 条数据已成功插入到MongoDB!")

# MONGO_DB.delete_many({})
# for root, dirs, files in os.walk("./new_data"):
#     for file in files:
#         if file.endswith(".csv"):
# # # CSV文件路径
# # csv_file_path = "./temporal_data/美国派前国防官员访台.csv"  # 替换为你的CSV文件路径

# # 获取event_name为CSV文件的前缀
#             event_name = os.path.splitext(os.path.basename(file))[0]
#             print(event_name)
#             if event_name not in events:
#                 continue
#             # MONGO_DB.delete_many({})
#             documents = []
#             # 读取CSV文件并插入到MongoDB
#             with open("./new_data/" + file,encoding='utf-8') as csvfile:
#                 reader = csv.reader(csvfile)
#                 for row in reader:
#                     if len(row) != 6: continue
#                     document = {
#                         "source_node": row[0],
#                         "source_attr": row[1],
#                         "edge": row[2],
#                         "target_node": row[3],
#                         "target_attr": row[4],
#                         "modify_time": row[5],
#                         "event_name": event_name
#                     }
#                     documents.append(document)

#                     # 如果达到批量大小，插入并清空列表
#                     if len(documents) == BATCH_SIZE:
#                         MONGO_DB.insert_many(documents)
#                         documents.clear()

#                 # 插入剩余的文档
#                 if documents:
#                     MONGO_DB.insert_many(documents)


#             print(f"{event_name}  数据已成功插入到eventPredictionEv!")