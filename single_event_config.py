#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单事件配置文件 - 最简单的使用方式
只需要修改文件开头的几个变量即可
"""

import os
import sys
import json
import subprocess
import itertools
from pathlib import Path

# 添加ElasticSearch目录到路径
sys.path.append('./ElasticSearch')

# ============================================================================
# 在这里填写您的事件信息 - 只需要修改这部分
# ============================================================================
#TODO：现在修改中方关系反向化后，原始数据文件也会更改，怀疑是main.py中方法重新写入，需要修改
#TODO：反向化check模式的sorted_output方法需要修改，现在绑定文件名，需要修改
# 事件名称
EVENT_NAME = "台积电获得英伟达AI芯片订单"

# 关键词实体列表
ENTITIES = ["台积电", "英伟达", "AI芯片", "半导体", "美国", "魏哲家"]

# 事件时间（可选，格式：YYYY-MM-DD）
EVENT_TIME = "2024-12-15"

# 运行模式：check / fit / init
MODE = "check"

# 时间跨度（天数）
TIMESPAN = "31"

# 是否需要从ES更新数据集
UPDATE_DATASET = False

# 是否需要处理中方关系反向化
PROCESS_CHINESE_RELATIONS = False

# 是否使用处理后的数据
USE_PROCESSED_DATA = True

# 中方相关实体关键词（用于识别中方相关的关系）
CHINESE_ENTITIES = ["中国","中方","中共","中央","主席","北京","习近平","华为","外交部"]

# 正向关系词汇
POSITIVE_RELATIONS = ["增进", "感到满意", "相信", "认为优秀", "欢迎", "认为有成就", "支持", "认可", "欣赏", "视作英雄", "喜欢", "认为可靠", "感谢", "认为热情"]

# 负向关系词汇
NEGATIVE_RELATIONS = ["担忧", "损害", "质疑", "感到不满", "认为非法", "认为恐怖", "威胁", "攻击", "认为缺乏", "批评", "认为有威胁", "认为有危机", "认为有暴力", "认为犯罪", "认为违规", "认为失败"]
# ============================================================================
# 自动化流程函数
# ============================================================================

def load_or_create_label_file(label_file_path="tsmc_label.json"):
    """加载或创建标签文件"""
    if os.path.exists(label_file_path):
        with open(label_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_label_file(label_data, label_file_path="tsmc_label.json"):
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

def run_model_check(event_name, mode, timespan, update_dataset, csv_file_path):
    """运行模型检查"""
    try:
        print(f"开始运行模型检查，事件: {event_name}")
        
        # 导入并直接调用run_model_for_check的main函数
        import sys
        sys.path.append('.')
        
        # 动态导入并调用
        import run_model_for_check
        run_model_for_check.main(event_name, mode, timespan, update_dataset, csv_file_path)
        
        print(f"模型检查完成: {event_name}")
        return True
            
    except Exception as e:
        print(f"运行模型检查时出错: {e}")
        return False

def check_csv_file_exists(event_name):
    """检查CSV文件是否存在"""
    csv_path = f"tsmc_es_data/{event_name}.csv"
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

def main():
    """主函数"""
    print("=" * 60)
    print("单事件自动化处理流程")
    print("=" * 60)
    print(f"事件名称: {EVENT_NAME}")
    print(f"关键词实体: {ENTITIES}")
    print(f"事件时间: {EVENT_TIME}")
    print(f"运行模式: {MODE}")
    print(f"时间跨度: {TIMESPAN}")
    print("=" * 60)
    
    # 确保数据目录存在
    os.makedirs("tsmc_es_data", exist_ok=True)
    if(UPDATE_DATASET):
        # 步骤1: 更新getGraphformEs.py配置
        print("\n步骤1: 更新ES配置...")
        if not update_getGraphformEs_config(EVENT_NAME, ENTITIES):
            print("配置更新失败，退出")
            return
        
        # 步骤2: 运行getGraphformEs.py获取数据
        print("\n步骤2: 从ES获取数据...")
        if not run_getGraphformEs():
            print("ES数据获取失败，退出")
            return
        
        # 步骤3: 检查CSV文件是否生成
        print("\n步骤3: 检查数据文件...")
        if not check_csv_file_exists(EVENT_NAME):
            print(f"未找到数据文件: tsmc_es_data/{EVENT_NAME}.csv")
            print("请检查ES查询是否成功")
            return

        # 步骤4: 生成标签并更新标签文件
        print("\n步骤4: 生成标签...")
        label_data = load_or_create_label_file()
        
        # 生成两两组合的标签
        tags = generate_tags_from_entities(ENTITIES)
        
        # 更新标签数据
        label_data[EVENT_NAME] = {
            "time": EVENT_TIME,
            "tags": tags
        }
        
        # 保存标签文件
        save_label_file(label_data)
        print(f"已生成 {len(tags)} 个标签")
    
    # 检查是否需要处理中方关系
    should_process_chinese_relations = PROCESS_CHINESE_RELATIONS
    
    if USE_PROCESSED_DATA:
        # 检查处理后的数据是否存在
        if not check_csv_file_exists(f"{EVENT_NAME}_processed"):
            print(f"未找到处理后的数据文件: tsmc_es_data/{EVENT_NAME}_processed.csv")
            should_process_chinese_relations = True
    
    # 步骤5: 处理中方关系反向化（可选）
    if should_process_chinese_relations:
        print("\n步骤5: 处理中方关系反向化...")
        original_csv_path = f"tsmc_es_data/{EVENT_NAME}.csv"
        processed_csv_path = f"tsmc_es_data/{EVENT_NAME}_processed.csv"
        
        try:
            processed_count = reverse_chinese_relations(original_csv_path, processed_csv_path, CHINESE_ENTITIES, POSITIVE_RELATIONS, NEGATIVE_RELATIONS)
            if processed_count > 0:
                print(f"中方关系反向化处理成功，处理了 {processed_count} 条记录")
                print(f"处理后的文件: {processed_csv_path}")
                # 使用处理后的文件进行后续操作
                final_csv_path = processed_csv_path
            else:
                print("没有发现中方相关关系，使用原始文件")
                final_csv_path = original_csv_path
        except Exception as e:
            print(f" 中方关系反向化处理失败: {e}")
            print("使用原始文件继续处理")
            final_csv_path = original_csv_path
    else:
        print("\n步骤5: 跳过中方关系反向化处理")
        final_csv_path = f"tsmc_es_data/{EVENT_NAME}.csv"

    if not USE_PROCESSED_DATA:
        final_csv_path = f"tsmc_es_data/{EVENT_NAME}.csv"

    
    # 步骤6: 运行模型检查
    print("\n步骤6: 运行模型检查...")
    if not run_model_check(EVENT_NAME, MODE, TIMESPAN, UPDATE_DATASET, final_csv_path):
        print("模型检查失败")
        return
    
    print("\n" + "=" * 60)
    print("自动化事件处理流程完成!")
    print(f"事件: {EVENT_NAME}")
    print(f"原始数据文件: tsmc_es_data/{EVENT_NAME}.csv")
    if should_process_chinese_relations and final_csv_path != f"tsmc_es_data/{EVENT_NAME}.csv":
        print(f"处理后数据文件: {final_csv_path}")
    print(f"标签文件: tsmc_label.json")
    print(f"输出目录: tsmc_es_data/{EVENT_NAME}/")
    print("=" * 60)

if __name__ == "__main__":
    main() 