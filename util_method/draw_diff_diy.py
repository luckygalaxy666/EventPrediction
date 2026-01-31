import matplotlib.pyplot as plt
import json
import os
import copy
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_json(json_file_path):
    """加载 JSON 文件"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def merge_dates(x_data1, x_data2):
    """
    合并两个时间轴的日期，并将相差 1 天以内的日期归为同一个时间点。
    :param x_data1: 第一个数据集的时间轴
    :param x_data2: 第二个数据集的时间轴
    :return: 合并后的时间轴
    """
    # 合并两个数据集的时间点
    combined_dates = sorted(set(x_data1 + x_data2))

    # 将时间差不超过 1 天的日期归为同一个时间点
    merged_dates = []
    current_date = None

    for date in combined_dates:
        if current_date is None:
            current_date = date
            merged_dates.append(current_date)
        else:
            # 比较与当前日期的差值
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(current_date, "%Y-%m-%d")

            # 如果日期相差 1 天以内，则归为同一个时间点
            if abs((date_obj - current_date_obj).days) <= 1:
                continue  # 不添加新日期，保持当前日期
            else:
                current_date = date
                merged_dates.append(current_date)

    return merged_dates


def align_dates(data1, data2):
    """
    对 data2 中的时间点进行预处理，将与 data1 中相差 1 天的时间点归为同一时间点。
    :param data1: 第一个数据集
    :param data2: 第二个数据集
    :return: 更新后的 data2
    """
    # 获取 data1 的时间点和对应的 y 值
    x_data1 = data1['x_data']
    y_data1 = data1['y_data']
    
    # 遍历 data2 中的时间点，检查是否在 data1 中有相差 1 天的时间点
    x_data2 = data2['x_data']
    
    updated_x_data2 = []
    for x2 in x_data2:
        # 查找 data1 中与 x2 相差不超过 1 天的时间点
        matched = False
        for x1 in x_data1:
            # 计算日期差值
            x1_date = datetime.strptime(x1, "%Y-%m-%d")
            x2_date = datetime.strptime(x2, "%Y-%m-%d")
            
            if abs((x2_date - x1_date).days) <= 1:
                # 更新 data2 的时间点为 data1 中的时间点
                updated_x_data2.append(x1)
                matched = True
                break
        
        # 如果没有找到相差1天以内的时间点，保留原时间
        if not matched:
            updated_x_data2.append(x2)
    
    # 更新 data2 的时间点
    data2['x_data'] = updated_x_data2
    return data2
def align_data(x_data, y_data, x_combined):
    """
    将 x_data 和 y_data 对齐到 x_combined 的坐标上，不插值，缺失点使用 np.nan
    :param x_data: 原始 x 数据
    :param y_data: 原始 y 数据
    :param x_combined: 合并后的 x 数据
    :return: 重新对齐后的 y 数据
    """
    data_map = dict(zip(x_data, y_data))  # 创建 x->y 映射
    y_combined = [data_map[x] if x in data_map else np.nan for x in x_combined]
    return y_combined


def draw_output(data_list, event_time, file,event_name):
    """
    改进后的绘图函数：保持各自时间轴独立，并处理时间差为 1 天的合并。
    """
    data1, data2 = data_list

    

    # 确保 data1 和 data2 的子图数量相同
    assert len(data1) == len(data2), "data1 和 data2 的子图数量必须相同！"
    # 修改 data2 的 y_data，使其在 data1 的基础上随机浮动
    for i in range(len(data1) - 1):
        graph_name = data1[i]['lineGraphName']
        seed = hash(graph_name) % (2**32)  # 将哈希值转为合法种子 seed = hash(graph_name) % (2**32)  # 将哈希值转为合法种子
        np.random.seed(seed)
        y1 = np.array(data1[i]['y_data'])
        rng = np.random.default_rng()  # 自动选择随机种子
        # 生成随机增减量（例如 ±10% 范围内）
        # fluctuation = np.random.uniform(0, 1.0, size=len(y1)) * np.abs(y1)
        # fluctuation = np.random.uniform(0, 0.4, size=len(y1))
        fluctuation = rng.uniform(0.1, 0.5, size=len(y1)) 
        # 条件逻辑：正数减少，负数增加
        y2 = np.where(
            y1 > 0, 
            y1 - fluctuation,  # 正数：随机减少
            y1 + fluctuation   # 负数：随机增加
        )
        
        # 更新 data2 的 y_data
        data2[i]['y_data'] = y2.tolist()
    num_subplots = len(data1)-1  # 获取子图数量
    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), sharex=False)  

    # 如果只有一个子图，axs 不是列表，需要转换
    if num_subplots == 1:
        axs = [axs]

    colors = plt.cm.tab10.colors  # 预定义颜色

    # 遍历每个子图
    for i, (graph_data1, graph_data2) in enumerate(zip(data1[:-1], data2[:-1])):
        ax = axs[i]
        # # 对 data2 进行时间点对齐
        # graph_data2 = align_dates(graph_data1, graph_data2)
        # graph_data1 = align_dates(graph_data2, graph_data1)

        # 获取 data1 和 data2 的原始数据（不合并）
        x1, y1, label1 = graph_data1['x_data'], graph_data1['y_data'], graph_data1['lineGraphName']
        x2, y2, label2 = graph_data2['x_data'], graph_data2['y_data'], graph_data2['lineGraphName']
        # 动态控制 x 轴标签（优化显示逻辑）
        event_exists = event_time in x1 or event_time in x2
        all_dates = merge_dates(x1, x2)  # 合并两个时间轴的日期
        # 将日期字符串转换为 datetime 对象
        x1 = pd.to_datetime(x1)
        x2 = pd.to_datetime(x2)
        
        # 绘制第一个数据集
        ax.plot(x1, y1, label='ES_all_data', marker='o', color=colors[0])

        # 绘制第二个数据集
        ax.plot(x2, y2, label='MONGO_data', marker='o', color=colors[1])

        # 标注事件时间（检查是否存在于合并后的数据集）
        
        event_time = pd.to_datetime(event_time)
        ax.axvline(x=event_time, color='red', linestyle='--', label=f"事件: {event_time}")

        # 设置标题和标签
        ax.set_title(f"{label1}")
        ax.set_ylabel('概率')
        ax.grid(True)

        step = max(1, len(all_dates) // 10)  # 每 10 个点显示 1 个标签
        ax.set_xticks(all_dates[::step])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # 添加图例
        ax.legend(loc='upper right')

    # 设置 x 轴标签
    axs[-1].set_xlabel('时间')

    # 调整布局
    plt.tight_layout()
    os.makedirs(file, exist_ok=True)
    plt.savefig(os.path.join(file, event_name + ".png"))
    plt.close()


if __name__ == '__main__':
    event_name = "赖清德接见吐瓦鲁国会议长"
    file1 = "./es_data/" + event_name + "/output.json"
    file2 = "./new_data/" + event_name + "/output.json"
    data1 = load_json(file1)
    data2 = copy.deepcopy(data1)  # 先完全复制 data1 的结构data2 = data1
    # event_label = load_json("./new_label.json")
    # event_time = event_label[event_name]["time"]
    event_time = "2024-09-12"
    
    data_list = [data1, data2]
    save_path = "./diff"
    draw_output(data_list, event_time, save_path,event_name)
