#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从ElasticSearch中提取所有relation_fixed中的实体并去重保存
"""

from elasticsearch import Elasticsearch
import json
from pathlib import Path

def connect_elasticsearch(host='http://121.48.163.69:45696'):
    """
    连接到 Elasticsearch 实例。
    
    参数:
        host (str): Elasticsearch 的连接地址。
    
    返回:
        Elasticsearch: Elasticsearch 客户端实例。
    """
    es = Elasticsearch(host)
    if not es.ping():
        raise ConnectionError("无法连接到 Elasticsearch 实例。请检查连接地址和 Elasticsearch 是否正在运行。")
    return es

def fetch_all_documents_with_relations(es, index_name, scroll='2m', size=1000):
    """
    使用 Scroll API 获取所有包含relation_fixed字段的文档。
    
    参数:
        es (Elasticsearch): Elasticsearch 客户端实例。
        index_name (str): 索引名称。
        scroll (str): Scroll 上下文保持时间。
        size (int): 每批次获取的文档数量。
    
    返回:
        list: 所有包含relation_fixed的文档列表。
    """
    # 首先尝试获取所有文档，然后过滤包含relation_fixed的
    query_body = {
        "query": {
            "match_all": {}
        }
    }
    
    # 初始搜索请求
    response = es.search(
        index=index_name,
        body=query_body,
        scroll=scroll,
        size=size,
        track_total_hits=True
    )

    scroll_id = response.get('_scroll_id')
    hits = response.get('hits', {}).get('hits', [])
    docs = hits

    print(f"初始获取文档数: {len(hits)}")

    # 使用 Scroll API 获取剩余文档
    while True:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll)
        scroll_id = response.get('_scroll_id')
        hits = response.get('hits', {}).get('hits', [])
        if not hits:
            break
        docs.extend(hits)
        print(f"累计获取文档数: {len(docs)}")

    # 清理 scroll 上下文
    es.clear_scroll(scroll_id=scroll_id)

    print(f"总命中数: {len(docs)}")
    return docs

def extract_entities_from_relations(docs):
    """
    从文档的relation_fixed字段中提取所有实体。
    
    参数:
        docs (list): 文档列表。
    
    返回:
        set: 去重后的实体集合。
    """
    entities = set()
    docs_with_relations = 0
    
    for doc in docs:
        source = doc.get('_source', {})
        
        # 尝试多种可能的字段名
        relation_fixed = source.get('relations', [])
            
        docs_with_relations += 1
        
        for relation in relation_fixed:
            # 提取头实体和尾实体
            first_entity = relation.get('firstEntity', '').strip()
            second_entity = relation.get('secondEntity', '').strip()
            
            # 添加到实体集合中（自动去重）
            if first_entity:
                entities.add(first_entity)
            if second_entity:
                entities.add(second_entity)
    
    print(f"包含关系数据的文档数: {docs_with_relations}")
    return entities

def save_entities_to_txt(entities, output_file):
    """
    将实体列表保存到txt文件中。
    
    参数:
        entities (set): 实体集合。
        output_file (str): 输出文件路径。
    """
    # 确保目录存在
    file_path = Path(output_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 将实体排序后写入文件
    sorted_entities = sorted(entities)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            f.write(entity + '\n')
    
    print(f"已保存 {len(entities)} 个实体到文件: {output_file}")

def main():
    """主函数"""
    print("=" * 60)
    print("从ElasticSearch提取所有relation_fixed中的实体")
    print("=" * 60)
    
    # ES配置（从getGraphformEs.py中获取）
    es_host = 'http://121.48.163.69:45696'
    index_name = 'tsmcnews'
    output_file = 'all_entities.txt'
    
    try:
        # 步骤1: 连接到ElasticSearch
        print("步骤1: 连接到ElasticSearch...")
        es = connect_elasticsearch(es_host)
        print(f"✅ 成功连接到: {es_host}")
        
        # 检查索引是否存在
        if not es.indices.exists(index=index_name):
            print(f"❌ 索引 '{index_name}' 不存在")
            print("可用的索引:")
            indices = es.indices.get_alias().keys()
            for idx in sorted(indices):
                print(f"  - {idx}")
            return False
        
        
        # 步骤2: 获取所有包含relation_fixed的文档
        print("\n步骤2: 获取包含relation_fixed的文档...")
        docs = fetch_all_documents_with_relations(es, index_name)
        print(f"✅ 获取到 {len(docs)} 个文档")
        
        # 调试：显示前几个文档的字段结构
        if docs:
            print("\n调试信息 - 前3个文档的字段结构:")
            for i, doc in enumerate(docs[:3]):
                source = doc.get('_source', {})
                print(f"\n文档 {i+1} 的字段:")
                for key in source.keys():
                    value = source[key]
                    if isinstance(value, list):
                        print(f"  {key}: [列表，长度: {len(value)}]")
                    elif isinstance(value, dict):
                        print(f"  {key}: [字典，键: {list(value.keys())}]")
                    else:
                        print(f"  {key}: {type(value).__name__}")
        else:
            print("⚠️ 没有获取到任何文档，请检查索引名称和连接")
        
        # 步骤3: 提取所有实体
        print("\n步骤3: 提取实体...")
        entities = extract_entities_from_relations(docs)
        print(f"✅ 提取到 {len(entities)} 个唯一实体")
        
        # 步骤4: 保存到文件
        print("\n步骤4: 保存实体到文件...")
        save_entities_to_txt(entities, output_file)
        
        # 步骤5: 显示统计信息
        print("\n" + "=" * 60)
        print("提取完成!")
        print("=" * 60)
        print(f"ES地址: {es_host}")
        print(f"索引名称: {index_name}")
        print(f"文档总数: {len(docs)}")
        print(f"实体总数: {len(entities)}")
        print(f"输出文件: {output_file}")
        
        # 显示前10个实体作为示例
        sorted_entities = sorted(entities)
        print(f"\n前10个实体示例:")
        for i, entity in enumerate(sorted_entities[:10], 1):
            print(f"  {i:2d}. {entity}")
        
        if len(sorted_entities) > 10:
            print(f"  ... 还有 {len(sorted_entities) - 10} 个实体")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 