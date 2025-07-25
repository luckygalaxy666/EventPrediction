import random
import pandas as pd

# 设置固定随机种子
random.seed(42)
def find_edges_with_entities(csv_file, entities):
    """
    从CSV文件中查找包含指定实体的所有边
    
    参数:
        csv_file (str): CSV文件路径
        entities (list): 要查找的实体名称列表
        
    返回:
        DataFrame: 包含匹配实体的所有边
    """
    # label_id = {"支持": 1, "喜欢": 2, "认为优秀": 3, "视作英雄": 4,
    #         "相信": 5, "欢迎": 6, "认为热情": 7, "感谢": 8, "认可": 9, "主导": 10, "增进": 11, "认为可靠": 12,
    #         "认为有成就": 13,
    #         "感到满意": 14, "欣赏": 15, "认为有威胁": 16, "威胁": 17, "认为恐怖": 18, "认为缺乏": 19, "攻击": 20, "批评": 21,
    #         "认为违规": 22, "认为有暴力": 23, "质疑": 24, "认为失败": 25, "认为犯罪": 26, "认为非法": 27, "损害": 28,
    #         "感到不满": 29,
    #         "担忧": 30, "认为有危机": 31, '党派': 32, '职务': 33, '竞争': 34, '机构': 35, '上级': 36}
    positive_labels = [
    "支持", "喜欢", "认为优秀", "视作英雄", "相信", "欢迎", 
    "认为热情", "感谢", "认可", "主导", "增进", "认为可靠", 
    "认为有成就", "感到满意", "欣赏"
]
    negative_labels = [
    "认为有威胁", "威胁", "认为恐怖", "认为缺乏", "攻击", "批评", 
    "认为违规", "认为有暴力", "质疑", "认为失败", "认为犯罪", 
    "认为非法", "损害", "感到不满", "担忧", "认为有危机"]
    # 读取CSV文件
    df = pd.read_csv(csv_file, header=None, names=[
        'head_entity', 'head_type', 'relation', 
        'tail_entity', 'tail_type', 'time'
    ])
    
    # # 查找头实体或尾实体在给定列表中的边
    # mask = df['head_entity'].isin(entities) | df['tail_entity'].isin(entities)
    # result = df[mask]
    result = {}
    for edge in df.itertuples(index=False):
        head_entity = edge.head_entity
        tail_entity = edge.tail_entity
        if head_entity in entities and tail_entity in entities:
            if(head_entity == tail_entity):
                continue
            # 如果头实体并且尾实体在目标实体列表中，则添加到结果中
            key = (f"{head_entity} - {tail_entity}")
            reversed_key = (f"{tail_entity} - {head_entity}")
            # 如果已经存在反向关系，则使用反向关系的键
            if reversed_key in result:
                key = reversed_key
            if key not in result:
                result[key] = {"positive_labels": [], "negative_labels": [], "attitude": 0}
            # 添加关系和倾向数量
            if edge.relation in positive_labels:
                if edge.relation not in result[key]["positive_labels"]:  # 修正：用字符串键访问
                    result[key]["positive_labels"].append(edge.relation)
                result[key]["attitude"] += 1
            elif edge.relation in negative_labels:
                if edge.relation not in result[key]["negative_labels"]:  # 修正：用字符串键访问
                    result[key]["negative_labels"].append(edge.relation)
                result[key]["attitude"] -= 1
            
    final_result = []
    for key, value in result.items():
        head_entity, tail_entity = key.split(' - ')
        if value['attitude'] > 0:
            relation =  value['positive_labels'][random.randint(0, len(value['positive_labels'])-1)]
            final_result.append({
                'head_entity': head_entity,
                'relation': relation,
                'tail_entity': tail_entity,
                # "attitude": value['attitude']
            })
        elif value['attitude'] < 0:
            relation =  value['negative_labels'][random.randint(0, len(value['negative_labels'])-1)]
            final_result.append({
                'head_entity': head_entity,
                'relation': relation,
                'tail_entity': tail_entity,
                # "attitude": value['attitude']
            })
    final_result = pd.DataFrame(final_result)

    # 去掉第一行
    if not final_result.empty:
        final_result = final_result.drop_duplicates(subset=['head_entity', 'relation', 'tail_entity'])
    return final_result

# 示例使用
if __name__ == "__main__":
    event_name = "台积夺单OpenAI自研晶片"
    # event_name = "台积电拿下特斯拉辅助驾驶芯片大单"  # 替换为你要查找的事件名称
    # 替换为你的CSV文件路径
    path = "./lrd_test/es_all_data/" 
    csv_path = path + event_name + ".csv"
    
    # 要查找的实体列表
    # "台积电拿下特斯拉辅助驾驶芯片大单": ["台积电", "特斯拉", "芯片", "半导体", "美国", "魏哲家","三星"],
    target_entities = ["台积电", "OpenAI", "晶片","马斯克","半导体", "美国", "魏哲家","特朗普"]  # 替换为你要查找的实体
    # target_entities = ["台积电", "特斯拉", "芯片", "半导体", "美国", "魏哲家","三星"]  # 替换为你要查找的实体
    # 查找并打印结果
    matching_edges = find_edges_with_entities(csv_path, target_entities)
    print(f"找到 {len(matching_edges)} 条包含指定实体的边:")
    print(matching_edges)
    
    # 可选：将结果保存到新CSV文件
    new_path =path + event_name+"_atomic.csv" 
    matching_edges.to_csv(new_path, index=False)
    print("结果已保存到" + new_path)