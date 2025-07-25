from py2neo import Graph, Node, Relationship
import pandas as pd

# 读取三元组数据（示例文件格式：头实体,关系,尾实体）
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=["head", "relation", "tail"])
    return df.drop_duplicates()

# 连接 Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "lrd2001226"))  # 替换你的密码
graph.delete_all()  # 清空现有数据（谨慎操作！）

# 导入数据到 Neo4j
def import_to_neo4j(file_path):
    df = load_data(file_path)
    for _, row in df.iterrows():
        head = Node("Entity", name=row["head"])  # 创建头节点
        tail = Node("Entity", name=row["tail"])  # 创建尾节点
        rel = Relationship(head, row["relation"], tail)  # 创建关系
        graph.merge(head, "Entity", "name")  # 避免重复节点
        graph.merge(tail, "Entity", "name")
        graph.create(rel)

# 执行导入
file_path = "./lrd_test/es_all_data/台积电拿下特斯拉辅助驾驶芯片大单_atomic.txt"
import_to_neo4j(file_path)
print("数据已导入 Neo4j！")