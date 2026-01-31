"""
输入文件格式说明：
每一行表示一个三元组，使用英文逗号分隔，格式如下：
    头实体, 关系, 尾实体
示例：
    台积电,竞争,三星
    台积电,合作,苹果
"""

import networkx as nx
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def read_data_from_file(file_path):
    """读取三元组数据文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                data.append((parts[0].strip(), parts[2].strip(), parts[1].strip()))  # 修正顺序：头实体,关系,尾实体
        return data

# 文件路径
file_path = './lrd_test/es_all_data/台积电拿下特斯拉辅助驾驶芯片大单_atomic.txt'
output_path = './knowledge_graph_directed.png'  # 输出图片路径

# 读取数据
data = read_data_from_file(file_path)

# 创建有向图
G = nx.DiGraph()  # 关键修改：使用 DiGraph 代替 Graph
for head, rel, tail in data:
    G.add_edge(head, tail, label=rel)

# 计算布局（有向图推荐使用 spring_layout 或 shell_layout）
pos = nx.spring_layout(G, seed=42, k=0.8)  # k 控制节点间距

# 设置画布大小
plt.figure(figsize=(16, 16))

# 绘制节点
nx.draw_networkx_nodes(
    G, pos, 
    node_size=3000, 
    node_color='skyblue', 
    alpha=0.8,
    edgecolors='black',  # 节点边框
    linewidths=2
)

# 绘制有向边（关键修改：添加箭头）
nx.draw_networkx_edges(
    G, pos, 
    width=2, 
    alpha=0.6, 
    edge_color='gray',
    arrows=True,                  # 启用箭头
    arrowstyle='-|>',            # 箭头样式
    arrowsize=20,                # 箭头大小
    connectionstyle='arc3,rad=0.1'  # 边弯曲度（可选）
)

# 绘制节点标签
nx.draw_networkx_labels(
    G, pos, 
    font_size=20, 
    font_family='sans-serif',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  # 标签背景
)

# 绘制边标签
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(
    G, pos, 
    edge_labels=edge_labels, 
    font_size=20, 
    font_family='sans-serif',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.5)  # 标签背景
)

# 添加标题
plt.title("知识图谱（有向图）", fontsize=25, pad=20)

# 保存图片
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()

print(f"有向知识图谱已保存至: {output_path}")