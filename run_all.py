import csv
import os
import subprocess
from pathlib import Path
import json 

# 加载并读取JSON数据的函数
def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def run_command(dataset_file,mode,label,time,checkpoint):
    command = [
        "python", "main.py",
        "--dataset", dataset_file,
        "--model", "regcn",
        "--weight-decay", "1e-5",
        "--early-stop", "3",
        "--train",
        "--label", str(label),
        "--timespan","30",
        "--time",str(time)
    ]
    if mode == 'fit' or mode == 'check':
        command+=["--checkpoint",checkpoint]
    
    
    print(f"Running command for {dataset_file}...")
    subprocess.run(command)
    print(f"Finished running for {dataset_file}\n")

def cal_accuracy(result_file):
    correct_count = 0
    total_count = 0 
    with open(result_file, 'r',encoding='utf-8') as f:
        for line in f:
            if "预测正确" in line:
                correct_count += 1
            if line != '\n':
                total_count += 1
    if total_count > 0:
        accuracy = correct_count / total_count
        with open(result_file,'a') as f:
            f.write(f"正确数：{correct_count}\n")
            f.write(f"总数：{total_count}\n")
            f.write(f"准确率：{accuracy}\n")


def main(folder_path,mode,flag):
    
    # 加载JSON 标签 数据
    data = load_json("label.json")
    # 遍历文件夹，找到所有的 .csv 文件
    failed = []
    # for root, dirs, files in os.walk(folder_path):
    #     # for dir in dirs:
    # #     #     if dir.endswith("next"):
    # #     #         shutil.rmtree(os.path.join(root,dir))
    #     for file in files:
    #         if file.endswith("4month.csv"):
    file = '一中决议_4month.csv'
    if(True):
                prefix = Path(file).stem
                event = prefix.split('_')[0]
                try:
                    label = data[event]["tags"]
                    time  = data[event]["time"]
                except:
                    failed.append(event)
                    # continue
                dataset_file = os.path.join(folder_path, prefix)

                os.makedirs(dataset_file, exist_ok=True)
                #删除多余模型
                # ff = os.path.join(dataset_file+'/checkpoint/regcn')
                # for _,dds,_ in os.walk(ff):
                #     for dd in dds:
                #         if dd.startswith('20240'):
                #             shutil.rmtree(os.path.join(ff,dd))

                with open (dataset_file +'/predict_compare.txt', flag,encoding='utf-8') as f:
                    f.write(f"{event} \n")
                if(flag == 'w'):
                    with open (dataset_file +'/predict_prev_score.txt', flag,encoding='utf-8') as f:
                        f.write(f"{event}\n")
                    with open (dataset_file +'/predict_now_score.txt', flag,encoding='utf-8') as f:
                        f.write(f"{event}\n")

                if mode == 'check':
                    checkpoint = 'base_model'
                elif mode  == 'fit':
                    checkpoint = time
                else: 
                    checkpoint = "111"
                run_command(dataset_file,mode,label,time,checkpoint)
    print(f"Failed to find label for {failed}")
    # cal_accuracy(folder_path+'/result/result_1.txt')
    
            


if __name__ == "__main__":
    
    # args = parser.parse_args()
    # main(args.folder)
    folder = '/home/code/KGMH-MAIN/lrd_test/new_data'
    # mode = 'init'
    # main(folder,mode,flag='w')
    
     
    flag = 'a'
    mode  = 'check'
    main(folder,mode,flag)
    # main(folder,'fit','a')
    # split_data(folder)