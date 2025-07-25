import os
import csv
import pandas as pd

def count_csv_rows(file_path):
    """统计 CSV 文件的行数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count

def count_csv_in_directory(directory):
    """统计目录中所有 CSV 文件的行数（不包括子目录）"""
    csv_counts = {}
    # 只遍历当前目录，忽略子目录
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            row_count = count_csv_rows(file_path)
            csv_counts[file] = row_count
    return csv_counts

def save_results_to_excel(new_data_counts, es_data_counts, output_file):
    """将结果保存到 Excel 文件中"""
    # 获取所有文件名（去重）
    all_files = set(new_data_counts.keys()).union(set(es_data_counts.keys()))
    
    # 创建 DataFrame
    data = []
    for file_name in sorted(all_files):
        # 获取 new_data 和 es_data 中的行数，如果不存在则为 0
        new_data_count = new_data_counts.get(file_name, 0)
        es_data_count = es_data_counts.get(file_name, 0)
        data.append([file_name, new_data_count, es_data_count])
    
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=["文件名", "new_data行数", "es_data行数"])
    
    # 保存到 Excel 文件
    df.to_excel(output_file, index=False)

def main():
    # 定义目录路径
    new_data_dir = "./new_data"
    es_data_dir = "./es_data"
    output_file = "csv_row_counts.xlsx"

    # 统计 new_data 和 es_data 中的 CSV 文件行数
    new_data_counts = count_csv_in_directory(new_data_dir)
    es_data_counts = count_csv_in_directory(es_data_dir)

    # 将结果保存到 Excel 文件
    save_results_to_excel(new_data_counts, es_data_counts, output_file)
    print(f"结果已保存到 {output_file}")

if __name__ == '__main__':
    main()