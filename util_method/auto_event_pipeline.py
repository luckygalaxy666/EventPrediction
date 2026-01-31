#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化事件处理流程
整合从ElasticSearch获取数据、生成标签、运行预测模型的完整流程
"""

import os
import sys
import json
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
import argparse

# 添加ElasticSearch目录到路径
sys.path.append('./ElasticSearch')

def load_or_create_label_file(label_file_path="tsmc_label.json"):
    """
    加载或创建标签文件
    
    Args:
        label_file_path (str): 标签文件路径
        
    Returns:
        dict: 标签数据
    """
    if os.path.exists(label_file_path):
        with open(label_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_label_file(label_data, label_file_path="tsmc_label.json"):
    """
    保存标签文件
    
    Args:
        label_data (dict): 标签数据
        label_file_path (str): 标签文件路径
    """
    with open(label_file_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=4)
    print(f"标签文件已保存到: {label_file_path}")

def generate_tags_from_entities(entities):
    """
    从实体列表生成两两组合的标签
    
    Args:
        entities (list): 实体列表
        
    Returns:
        list: 标签列表
    """
    tags = []
    # 生成所有两两组合
    for entity1, entity2 in itertools.combinations(entities, 2):
        # 为每个组合生成两种关系：利好和不利好
        tags.append(f"{entity1} 利好 {entity2}")
        tags.append(f"{entity2} 利好 {entity1}")
        tags.append(f"{entity1} 不利好 {entity2}")
        tags.append(f"{entity2} 不利好 {entity1}")
    
    return tags

def update_getGraphformEs_config(event_name, part_key_entities):
    """
    更新getGraphformEs.py中的配置
    
    Args:
        event_name (str): 事件名称
        part_key_entities (list): 关键词实体列表
    """
    es_file_path = "ElasticSearch/getGraphformEs.py"
    
    if not os.path.exists(es_file_path):
        print(f"错误: 找不到文件 {es_file_path}")
        return False
    
    # 读取文件内容
    with open(es_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新事件名称
    content = content.replace(
        'event_name = "台积夺单OpenAI自研晶片"',
        f'event_name = "{event_name}"'
    )
    
    # 更新关键词实体
    entities_str = '["' + '", "'.join(part_key_entities) + '"]'
    content = content.replace(
        'part_key_entities = ["台积电", "OpenAI", "晶片","马斯克","半导体", "美国", "魏哲家","特朗普"]',
        f'part_key_entities = {entities_str}'
    )
    
    # 写回文件
    with open(es_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已更新 {es_file_path} 配置")
    return True

def run_getGraphformEs():
    """
    运行getGraphformEs.py获取ES数据
    
    Returns:
        bool: 是否成功
    """
    try:
        print("开始从ElasticSearch获取数据...")
        result = subprocess.run(
            ["python", "ElasticSearch/getGraphformEs.py"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("ES数据获取成功")
            return True
        else:
            print(f"ES数据获取失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"运行getGraphformEs.py时出错: {e}")
        return False

def run_model_check(event_name, mode='check', timespan='31', update_dataset=False):
    """
    运行模型检查
    
    Args:
        event_name (str): 事件名称
        mode (str): 运行模式
        timespan (str): 时间跨度
        update_dataset (bool): 是否更新数据集
        
    Returns:
        bool: 是否成功
    """
    try:
        print(f"开始运行模型检查，事件: {event_name}")
        
        # 导入并直接调用run_model_for_check的main函数
        import sys
        sys.path.append('.')
        
        # 动态导入并调用
        import run_model_for_check
        run_model_for_check.main(event_name, mode, timespan, update_dataset)
        
        print(f"模型检查完成: {event_name}")
        return True
            
    except Exception as e:
        print(f"运行模型检查时出错: {e}")
        return False

def check_csv_file_exists(event_name):
    """
    检查CSV文件是否存在
    
    Args:
        event_name (str): 事件名称
        
    Returns:
        bool: 文件是否存在
    """
    csv_path = f"tsmc_es_data/{event_name}.csv"
    return os.path.exists(csv_path)

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='自动化事件处理流程')
    parser.add_argument('--event_name', required=True, help='事件名称')
    parser.add_argument('--entities', required=True, nargs='+', help='关键词实体列表')
    parser.add_argument('--event_time', default='Unknown', help='事件时间 (YYYY-MM-DD格式)')
    parser.add_argument('--mode', default='check', choices=['check', 'fit', 'init'], help='运行模式')
    parser.add_argument('--timespan', default='31', help='时间跨度')
    parser.add_argument('--label_file', default='tsmc_label.json', help='标签文件路径')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("自动化事件处理流程开始")
    print(f"事件名称: {args.event_name}")
    print(f"关键词实体: {args.entities}")
    print(f"事件时间: {args.event_time}")
    print(f"运行模式: {args.mode}")
    print("=" * 50)
    
    # 步骤1: 更新getGraphformEs.py配置
    print("\n步骤1: 更新ES配置...")
    if not update_getGraphformEs_config(args.event_name, args.entities):
        print("配置更新失败，退出")
        return
    
    # 步骤2: 运行getGraphformEs.py获取数据
    print("\n步骤2: 从ES获取数据...")
    if not run_getGraphformEs():
        print("ES数据获取失败，退出")
        return
    
    # 步骤3: 检查CSV文件是否生成
    print("\n步骤3: 检查数据文件...")
    if not check_csv_file_exists(args.event_name):
        print(f"未找到数据文件: tsmc_es_data/{args.event_name}.csv")
        print("请检查ES查询是否成功")
        return
    
    print(f"数据文件已生成: tsmc_es_data/{args.event_name}.csv")
    
    # 步骤4: 生成标签并更新标签文件
    print("\n步骤4: 生成标签...")
    label_data = load_or_create_label_file(args.label_file)
    
    # 生成两两组合的标签
    tags = generate_tags_from_entities(args.entities)
    
    # 更新标签数据
    label_data[args.event_name] = {
        "time": args.event_time,
        "tags": tags
    }
    
    # 保存标签文件
    save_label_file(label_data, args.label_file)
    print(f"已生成 {len(tags)} 个标签")
    
    # 步骤5: 运行模型检查
    print("\n步骤5: 运行模型检查...")
    if not run_model_check(args.event_name, args.mode, args.timespan, False):
        print("模型检查失败")
        return
    
    print("\n" + "=" * 50)
    print("自动化事件处理流程完成!")
    print(f"事件: {args.event_name}")
    print(f"数据文件: tsmc_es_data/{args.event_name}.csv")
    print(f"标签文件: {args.label_file}")
    print(f"输出目录: tsmc_es_data/{args.event_name}/")
    print("=" * 50)

if __name__ == "__main__":
    main() 