#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于配置文件的自动化事件处理流程
用户可以在文件内直接填写事件信息，无需命令行输入
"""

import os
import sys
import json
import subprocess
import itertools
from pathlib import Path
from datetime import datetime

# 添加ElasticSearch目录到路径
sys.path.append('./ElasticSearch')

# ============================================================================
# 配置文件区域 - 请在这里填写您的事件信息
# ============================================================================

# 事件配置列表 - 可以配置多个事件
EVENT_CONFIGS = [
    {
        "event_name": "台积电获得英伟达AI芯片订单",
        "entities": ["台积电", "英伟达", "AI芯片", "半导体", "美国", "魏哲家"],
        "event_time": "2024-12-15",
        "mode": "check",
        "timespan": "31"
    },
    {
        "event_name": "赖清德访问美国",
        "entities": ["赖清德", "美国", "台湾", "拜登", "中国"],
        "event_time": "2024-11-20",
        "mode": "check",
        "timespan": "31"
    },
    {
        "event_name": "台积电在德国建厂",
        "entities": ["台积电", "德国", "欧洲", "半导体", "魏哲家"],
        "event_time": "2024-10-25",
        "mode": "check",
        "timespan": "31"
    }
]

# 全局配置
GLOBAL_CONFIG = {
    "label_file": "tsmc_label.json",  # 标签文件路径
    "data_dir": "tsmc_es_data",       # 数据目录
    "es_host": "http://121.48.163.69:45696",  # ES连接地址
    "es_index": "tsmcnews",           # ES索引名称
    "start_date": "2024-06-01",       # 查询开始日期
    "end_date": "2025-02-01",         # 查询结束日期
    "min_score": 10                   # 最低得分阈值
}

# 中方关系反向化配置
CHINESE_RELATIONS_CONFIG = {
    "enabled": True,  # 是否启用中方关系反向化
    "chinese_entities": ["中国","中方","中共","中央","主席","北京","习近平","华为","外交部"],  # 中方相关实体关键词
    "positive_relations": ["增进", "感到满意", "相信", "认为优秀", "欢迎", "认为有成就", "支持", "认可", "欣赏", "视作英雄", "喜欢", "认为可靠", "感谢", "认为热情"],  # 正向关系词汇
    "negative_relations": ["担忧", "损害", "质疑", "感到不满", "认为非法", "认为恐怖", "威胁", "攻击", "认为缺乏", "批评", "认为有威胁", "认为有危机", "认为有暴力", "认为犯罪", "认为违规", "认为失败"]  # 负向关系词汇
}

# 是否只处理第一个事件（用于测试）
PROCESS_FIRST_ONLY = False

# ============================================================================
# 自动化流程函数
# ============================================================================

def load_or_create_label_file(label_file_path):
    """加载或创建标签文件"""
    if os.path.exists(label_file_path):
        with open(label_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_label_file(label_data, label_file_path):
    """保存标签文件"""
    with open(label_file_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=4)
    print(f"标签文件已保存到: {label_file_path}")

def generate_tags_from_entities(entities):
    """从实体列表生成两两组合的标签"""
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
    """更新getGraphformEs.py中的配置"""
    es_file_path = "ElasticSearch/getGraphformEs.py"
    
    if not os.path.exists(es_file_path):
        print(f"错误: 找不到文件 {es_file_path}")
        return False
    
    # 读取文件内容
    with open(es_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式更新事件名称，匹配任何当前的事件名称
    import re
    content = re.sub(
        r'event_name = "[^"]*"',
        f'event_name = "{event_name}"',
        content
    )
    
    # 使用正则表达式更新关键词实体，匹配任何当前的实体列表
    entities_str = '["' + '", "'.join(part_key_entities) + '"]'
    content = re.sub(
        r'part_key_entities = \[[^\]]*\]',
        f'part_key_entities = {entities_str}',
        content
    )
    
    # 写回文件
    with open(es_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已更新 {es_file_path} 配置")
    print(f"  事件名称: {event_name}")
    print(f"  关键词实体: {part_key_entities}")
    return True

def run_getGraphformEs():
    """运行getGraphformEs.py获取ES数据"""
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
    """运行模型检查"""
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
    """检查CSV文件是否存在"""
    csv_path = f"{GLOBAL_CONFIG['data_dir']}/{event_name}.csv"
    return os.path.exists(csv_path)

def is_chinese_entity(entity, chinese_entities):
    """判断实体是否为中方相关实体（使用正则匹配）"""
    import re
    for chinese_entity in chinese_entities:
        if re.search(chinese_entity, entity):
            return True
    return False

def reverse_chinese_relations(csv_file_path, output_file_path, chinese_entities, positive_relations, negative_relations):
    """
    处理中方相关的四元组关系，将其改为反向倾向
    只有当头实体包含中方相关实体关键词时才处理
    
    Args:
        csv_file_path (str): 原始CSV文件路径
        output_file_path (str): 输出CSV文件路径
        chinese_entities (list): 中方相关实体关键词列表
        positive_relations (list): 正向关系词汇列表
        negative_relations (list): 负向关系词汇列表
    """
    import csv
    import random
    
    print(f"开始处理中方关系反向化...")
    print(f"输入文件: {csv_file_path}")
    print(f"输出文件: {output_file_path}")
    
    processed_count = 0
    total_count = 0
    
    with open(csv_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            total_count += 1
            
            if len(row) != 6:
                # 如果行格式不正确，直接写入
                writer.writerow(row)
                continue
            
            source_entity, source_type, relation, target_entity, target_type, date = row
            
            # 检查头实体（source_entity）是否为中方相关实体
            source_is_chinese = is_chinese_entity(source_entity, chinese_entities)
            
            if source_is_chinese:
                # 只有当头实体包含中方相关实体关键词时才处理
                new_relation = relation
                
                # 检查是否为正向关系
                is_positive = any(pos_rel in relation for pos_rel in positive_relations)
                # 检查是否为负向关系
                is_negative = any(neg_rel in relation for neg_rel in negative_relations)
                
                if is_positive:
                    # 正向关系改为负向关系
                    new_relation = random.choice(negative_relations)
                    processed_count += 1
                    print(f"  处理: {source_entity} {relation} {target_entity} -> {source_entity} {new_relation} {target_entity}")
                elif is_negative:
                    # 负向关系改为正向关系
                    new_relation = random.choice(positive_relations)
                    processed_count += 1
                    print(f"  处理: {source_entity} {relation} {target_entity} -> {source_entity} {new_relation} {target_entity}")
                else:
                    # 其他关系保持不变
                    new_relation = relation
                
                # 写入处理后的行
                writer.writerow([source_entity, source_type, new_relation, target_entity, target_type, date])
            else:
                # 头实体不包含中方相关实体关键词，直接写入原始行
                writer.writerow(row)
    
    print(f"中方关系反向化处理完成!")
    print(f"总记录数: {total_count}")
    print(f"处理记录数: {processed_count}")
    print(f"处理比例: {processed_count/total_count*100:.2f}%")
    
    return processed_count

def process_single_event(event_config):
    """处理单个事件"""
    event_name = event_config["event_name"]
    entities = event_config["entities"]
    event_time = event_config["event_time"]
    mode = event_config["mode"]
    timespan = event_config["timespan"]
    
    print("\n" + "=" * 60)
    print(f"开始处理事件: {event_name}")
    print(f"关键词实体: {entities}")
    print(f"事件时间: {event_time}")
    print(f"运行模式: {mode}")
    print("=" * 60)
    
    # 步骤1: 更新getGraphformEs.py配置
    print("\n步骤1: 更新ES配置...")
    if not update_getGraphformEs_config(event_name, entities):
        print("配置更新失败，跳过此事件")
        return False
    
    # 步骤2: 运行getGraphformEs.py获取数据
    print("\n步骤2: 从ES获取数据...")
    if not run_getGraphformEs():
        print("ES数据获取失败，跳过此事件")
        return False
    
    # 步骤3: 检查CSV文件是否生成
    print("\n步骤3: 检查数据文件...")
    if not check_csv_file_exists(event_name):
        print(f"未找到数据文件: {GLOBAL_CONFIG['data_dir']}/{event_name}.csv")
        print("请检查ES查询是否成功，跳过此事件")
        return False
    
    print(f"数据文件已生成: {GLOBAL_CONFIG['data_dir']}/{event_name}.csv")
    
    # 步骤4: 处理中方关系反向化（可选）
    final_csv_path = f"{GLOBAL_CONFIG['data_dir']}/{event_name}.csv"
    if CHINESE_RELATIONS_CONFIG["enabled"]:
        print("\n步骤4: 处理中方关系反向化...")
        original_csv_path = f"{GLOBAL_CONFIG['data_dir']}/{event_name}.csv"
        processed_csv_path = f"{GLOBAL_CONFIG['data_dir']}/{event_name}_processed.csv"
        
        try:
            processed_count = reverse_chinese_relations(
                original_csv_path, 
                processed_csv_path, 
                CHINESE_RELATIONS_CONFIG["chinese_entities"], 
                CHINESE_RELATIONS_CONFIG["positive_relations"], 
                CHINESE_RELATIONS_CONFIG["negative_relations"]
            )
            if processed_count > 0:
                print(f"✅ 中方关系反向化处理成功，处理了 {processed_count} 条记录")
                print(f"📁 处理后的文件: {processed_csv_path}")
                # 使用处理后的文件进行后续操作
                final_csv_path = processed_csv_path
            else:
                print("ℹ️ 没有发现中方相关关系，使用原始文件")
                final_csv_path = original_csv_path
        except Exception as e:
            print(f"⚠️ 中方关系反向化处理失败: {e}")
            print("使用原始文件继续处理")
            final_csv_path = original_csv_path
    else:
        print("\n步骤4: 跳过中方关系反向化处理")
    
    # 步骤5: 生成标签并更新标签文件
    print("\n步骤5: 生成标签...")
    label_data = load_or_create_label_file(GLOBAL_CONFIG["label_file"])
    
    # 生成两两组合的标签
    tags = generate_tags_from_entities(entities)
    
    # 更新标签数据
    label_data[event_name] = {
        "time": event_time,
        "tags": tags
    }
    
    # 保存标签文件
    save_label_file(label_data, GLOBAL_CONFIG["label_file"])
    print(f"已生成 {len(tags)} 个标签")
    
    # 步骤6: 运行模型检查
    print("\n步骤6: 运行模型检查...")
    # 如果启用了中方关系反向化，则跳过main.py中的中方处理
    update_dataset = CHINESE_RELATIONS_CONFIG["enabled"]
    if not run_model_check(event_name, mode, timespan, update_dataset):
        print("模型检查失败")
        return False
    
    print("\n" + "=" * 60)
    print(f"事件 '{event_name}' 处理完成!")
    print(f"数据文件: {GLOBAL_CONFIG['data_dir']}/{event_name}.csv")
    print(f"标签文件: {GLOBAL_CONFIG['label_file']}")
    print(f"输出目录: {GLOBAL_CONFIG['data_dir']}/{event_name}/")
    print("=" * 60)
    
    return True

def main():
    """主函数"""
    print("基于配置文件的自动化事件处理流程")
    print("=" * 60)
    print(f"配置的事件数量: {len(EVENT_CONFIGS)}")
    print(f"标签文件: {GLOBAL_CONFIG['label_file']}")
    print(f"数据目录: {GLOBAL_CONFIG['data_dir']}")
    print("=" * 60)
    
    # 确保数据目录存在
    os.makedirs(GLOBAL_CONFIG['data_dir'], exist_ok=True)
    
    # 处理事件
    success_count = 0
    total_count = len(EVENT_CONFIGS)
    
    if PROCESS_FIRST_ONLY:
        print("测试模式：只处理第一个事件")
        total_count = 1
    
    for i, event_config in enumerate(EVENT_CONFIGS):
        if PROCESS_FIRST_ONLY and i >= 1:
            break
            
        print(f"\n处理进度: {i+1}/{total_count}")
        
        try:
            if process_single_event(event_config):
                success_count += 1
        except Exception as e:
            print(f"处理事件 '{event_config['event_name']}' 时发生异常: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("处理完成总结")
    print("=" * 60)
    print(f"成功处理: {success_count}/{total_count} 个事件")
    
    if success_count == total_count:
        print("✅ 所有事件处理成功!")
    else:
        print(f"⚠️ 有 {total_count - success_count} 个事件处理失败")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 