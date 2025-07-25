from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers
import urllib.parse
from opencc import OpenCC

# MongoDB配置
mongo_host = "121.48.163.69"
mongo_port = 27018

username = "root"
password = "rootdb@"
auth_source = "admin"  # 使用的身份验证数据库

# 连接到MongoDB
# MONGO_DB = MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['2024-10-28-task']['T-tweets-keywords']
MONGO_DB = MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['2024-12-2-task']['TSMC-news']
# MONGO_DB = MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['2024-12-2-task']['TSMC-events']
# MONGO_DB = MongoClient(host="121.48.163.69", port=27018, username="root", password="rootdb@")['2024-07-15-google-search-task']['news_1']
# 连接到 Elasticsearch
try:
    es = Elasticsearch("http://121.48.163.69:45696")
    # 检查 Elasticsearch 连接
    if not es.ping():
        raise ValueError("Could not connect to Elasticsearch")
    print("Connected to Elasticsearch successfully!")
except Exception as e:
    print(f"Could not connect to Elasticsearch: {e}")
    exit(1)

# 创建 Elasticsearch 索引并设置分词器
index_name = "tsmcnews"
try:
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "ik_max_word"  # 使用 IK 分词器，需确保已安装
                        },
                        "relations": {
                            "type": "nested",
                            "properties": {
                                "firstEntity": {"type": "text"},
                                "secondEntity": {"type": "text"},
                                "relation": {"type": "text"}
                            }
                        },
                        "keywords": {
                            "type": "keyword"
                 
                        },
                        "date": {
                        "type": "date",
                        "format": "yyyy-MM-dd"
                        },

                    }
                }
            }
        )
        print(f"Index {index_name} created successfully.")
except Exception as e:
    print(f"Error creating index {index_name}: {e}")

# 初始化 OpenCC 转换器，选择从繁体到简体的转换模式
converter = OpenCC('t2s')

# 定义将数据从 MongoDB 转移到 Elasticsearch 的函数
def mongo_to_es():
    # 从 MongoDB 集合中获取所有文档
    cursor = MONGO_DB.find()

    actions = []
    for doc in cursor:
        # 提取所需字段
        content = doc.get("page_content_zh")
        if(content == None):
            content = doc.get("page_content")
        # content = doc.get("content")
        date =  doc.get("date1")
        relations = doc.get("relations", [])  # 默认值为空列表
        keywords = doc.get("keywords", "")  # 默认值为空字符串
        # unit_event = doc.get("unit_event", [])  # 默认值为空字符串
        if content == " This page has no content to extract" or content == None or date == None or relations == []:
            continue
        # 将 content 字段的繁体中文转换为简体中文
        content = converter.convert(content)
        # for event in unit_event:
        #     new_event = {}
        #     new_event["firstEntity"] = event["subject"]
        #     new_event["secondEntity"] = event["object"]
        #     new_event["relation"] = event["relation"]
        #     relations.append(new_event)
        # 将 relations 字段的繁体中文转换为简体中文
        for relation in relations:
            relation["firstEntity"] = converter.convert(relation["firstEntity"])
            relation["secondEntity"] = converter.convert(relation["secondEntity"])
            relation["relation"] = converter.convert(relation["relation"])
        data = {
            "content": content,
            "relations": relations,
            "date": date.strftime("%Y-%m-%d"),
            "keywords": converter.convert(keywords)
        }
        actions.append({
            "_index": index_name,
            "_id": str(doc["_id"]),  # 将 MongoDB 的 _id 转换为字符串并作为索引操作的参数
            "_source": data  # 不包含 _id 字段
        })
    
    # 批量插入到 Elasticsearch
    try:
        helpers.bulk(es, actions)
        print(f"Successfully indexed {len(actions)} documents.")
    except Exception as e:
        print(f"Error inserting documents into Elasticsearch: {e}")

# 执行数据转移函数
mongo_to_es()