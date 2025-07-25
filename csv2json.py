import csv
import json

# 定义空字典存储最终输出
output_data = {
    "links": [],
    "node": []
}

# 用来存储已处理的节点，避免重复
nodes_set = set()

csv_file_path = './new_data/蔡英文国庆发言1111.csv'  # 替换为实际的 CSV 文件路径
json_file_path = '22222.json'  # 输出的 JSON 文件路径

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    i=0
    for row in reader:
        i+=1
        if len(row) != 6:
            print("跳过无效行:", i)
            # print(row[:20])
            continue
        entity1, type1, relation, entity2, type2, date = row

        # 如果该节点尚未被处理过，添加节点信息
        if entity1 not in nodes_set:
            output_data["node"].append({
                "id": str(len(nodes_set) + 1),  # 自动生成ID，可以根据需要调整
                "name": entity1,
                "type": type1
            })
            nodes_set.add(entity1)
        
        if entity2 not in nodes_set:
            output_data["node"].append({
                "id": str(len(nodes_set) + 1),
                "name": entity2,
                "type": type2
            })
            nodes_set.add(entity2)

        # 查找实体的 id
        source_id = next(node["id"] for node in output_data["node"] if node["name"] == entity1)
        target_id = next(node["id"] for node in output_data["node"] if node["name"] == entity2)

        # 添加链接信息
        output_data["links"].append({
            "source_id": source_id,
            "target_id": target_id,
            "timeDate": date,
            "type": relation
        })

# 输出为 JSON 文件
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print("转换成功，文件已保存为:", json_file_path)