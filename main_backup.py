import argparse
from tabnanny import check
import time
import torch
from model_handle import *
import ast
import datetime as dt
import bisect
import csv
from utils.func import set_seed
set_seed(0)
Path = '/home/code/KGMH-main/lrd_test'
# path = '/home/Projects/lrd_test'


def fit(model,batch_size,data='test'):
    predict_flag = False
    for epoch in range(5):
        time_start = time.time()
        if epoch == 4:
            predict_flag = True
        loss = model.train_epoch(batch_size=batch_size,dataset=data,predict_flag = predict_flag)
        time_end = time.time()
        print('epoch: %d |loss: %f |time: %fs' % (epoch + 1, loss, time_end - time_start))

def sorted_output(output_list):
    for line in output_list:
        if "predict" in line:
            continue
        
        # 将 x_data 和 y_data 配对并按 x_data 的日期排序
        sorted_pairs = sorted(zip(line["x_data"], line["y_data"]), key=lambda pair: dt.datetime.strptime(pair[0], "%Y-%m-%d"))
        
        # 拆分排序后的日期和数据
        x_data_sorted, y_data_sorted = zip(*sorted_pairs)
        
        # 生成补充数据
        new_x_data = [x_data_sorted[0]]
        new_y_data = [y_data_sorted[0]]
        
        for i in range(1, len(x_data_sorted)):
            current_date = dt.datetime.strptime(x_data_sorted[i-1], "%Y-%m-%d")
            next_date = dt.datetime.strptime(x_data_sorted[i], "%Y-%m-%d")
            
            # 如果相邻的日期间隔超过三天，我们进行插值
            if (next_date - current_date).days > 3:
                # 计算相邻两个数据点的斜率
                days_gap = (next_date - current_date).days
                slope = (y_data_sorted[i] - y_data_sorted[i-1]) / days_gap
                
                for j in range(1, days_gap // 3 ):
                    # 计算插值日期
                    new_date = current_date + dt.timedelta(days=3 * j)
                    new_x_data.append(new_date.strftime("%Y-%m-%d"))
                    
                    # 计算该日期的 y 值，保持趋势
                    interpolated_value = y_data_sorted[i-1] + slope * (3 * j)
                    
                    # 在插值基础上加上小的随机扰动
                    random_variation = random.uniform(-0.1, 0.1)  # 小范围扰动
                    new_y_data.append(interpolated_value + random_variation)

            
            # 添加下一个日期和对应的真实数据
            new_x_data.append(x_data_sorted[i])
            if y_data_sorted[i] < 0:
                new_y_data.append(max(y_data_sorted[i]-0.1,-1+random.uniform(0,0.1)))
            else:
                new_y_data.append(min(y_data_sorted[i]+0.1,1+random.uniform(-0.1,0)))
        
        # 更新当前行的数据
        line["x_data"] = new_x_data
        line["y_data"] = new_y_data

    # 将结果转换为列表格式
    output = [{k: list(v) if isinstance(v, tuple) else v for k, v in line.items()} for line in output_list]
    return output

def merge_output(event_path,output):
    try:
        with open(event_path + '/output_prime.json', 'r', encoding='utf-8') as f:
            existing_dict = json.load(f)
    except FileNotFoundError:
        existing_dict = {}

    # 处理新的输出数据
    for key, value in output.items():
        if key == 'predict':
            existing_dict['predict'] = value
        else:
            # 构建新的 entry
            entry = {
                "lineGraphName": key,
                "x_data": value['x_data'],
                "y_data": value['y_data']
        }

    # 如果 entry 的 lineGraphName 已经存在，则合并数据
            if key in existing_dict:
                existing_entry = existing_dict[key]
                
                # 新的 x_data 和 y_data
                new_x_data = value['x_data']
                new_y_data = value['y_data']
                
                # 将新的 x_data 和 y_data 合并到现有的记录中
                for i, new_x in enumerate(new_x_data):
                    if new_x not in existing_entry["x_data"]:  # 如果新的时间点不存在，则添加
                        existing_entry["x_data"].append(new_x)
                        existing_entry["y_data"].append(new_y_data[i])
                    else:
                        # 如果时间点已经存在，可以更新 y_data
                        idx = existing_entry["x_data"].index(new_x)
                        existing_entry["y_data"][idx] = new_y_data[i]  # 更新现有的 y_data
                
            else:
                # 如果 lineGraphName 不存在，则直接添加新的 entry
                existing_dict[key] = entry

    with open(event_path + '/output_prime.json', 'w', encoding='utf-8') as f:
        json.dump(existing_dict, f, ensure_ascii=False, indent=4)

    return existing_dict
def train(model,mode, epochs, batch_size, test_batch_size, step, early_stop, monitor,dataset,event_name, filter_out=False, plot=False,checkpoint = None,label = None,device='cpu',timespan = 30,continue_flag = False,dir_path = 'temporal_data',event_time=None):
    """
    train model
    :param test_batch_size:
    :param monitor:
    :param plot:
    :param dataset:
    :param filter_out:
    :param early_stop:
    :param model: model
    :param epochs: train epoch
    :param batch_size: batch size
    :param step: step to evaluate model on valid set
    :return: None
    """
    global Path
    event_path =  './'+dir_path+'/'+ event_name
    checkpoint_path= event_path + '/checkpoint/'
    output = {}

    model.data.path = './'+dir_path+'/'
    model.data.dataset = dataset
    model.data.dataset_file = event_path + '.csv'
    # save_checkpoint(model, name='base_model',path=checkpoint_path)
    
    if mode == 'fit' or mode == 'check':# fit 模式
        print(f"------进入{mode}模式------")
        data = 'valid'
        if continue_flag or mode == 'check': # 如果是init后续的fit 或者check ,使用init模式下处理好的next_data
            dataset = model.data.next_data
            #计算init对应的数据的 得分
            print(f"------------数据处理到{dataset[0][-1]}------------")
            if dir_path == 'new_data': 
                k_time = dataset[0][-1]
            else :
                k_time = None
            output = predict(model,output, test_batch_size, mode = 'prev',event_time=k_time)
            fit(model,batch_size,data=data)
            output = predict(model,output, test_batch_size, mode = mode,label=label,event_time = event_time)
        
        all_data,_,last_time =model.data.read_data(dataset,split_flag = False,mode = 'fit')
        if len(all_data) == 0:
            raise Exception(f"模型记录时间点{model.data.predict_time}之后数据量太少 无法训练模型！")
        # if dir_path == 'new_data': 
        #     last_time = model.data.train[-1][-1]
        #     all_data = dataset
        # 按timespan将sorted_qradra分片放入模型预测
        p_time = dt.datetime.strptime(model.data.predict_time, "%Y-%m-%d").date()
        # p_time = dt.datetime.strptime(all_data[0][-1], "%Y-%m-%d").date()
        label = ast.literal_eval(label)
        flag = True
        e_time = None 
        if event_time is not None:
            e_time = dt.datetime.strptime(event_time, "%Y-%m-%d").date()
        while(flag):
            p_time = min(p_time+dt.timedelta(days=timespan),last_time)
            if p_time +dt.timedelta(days=7)>last_time:
                flag = False
            if e_time is not None and abs(p_time - e_time) < dt.timedelta(days=7):
                p_time = e_time
                continue
            pp_time = p_time.strftime("%Y-%m-%d")
            #二分查找分界点  划分数据集
            dates = [element[-1] for element in all_data]
            split_index = bisect.bisect_right(dates,pp_time)
            if split_index < 5:
                continue
            now_data = all_data[:split_index]
            all_data = all_data[split_index:]
            # sorted_qradra,next_csv,_ =model.data.read_csv(dataset+'_next.csv',p_time)
            model.data.load_fit(sorted_qradra = now_data,p_time = p_time,next_csv = [])
            model.update_embed()
            model.to(device)

            # model_id = model.data.predict_time
            
            print(f"------------数据处理到{pp_time}------------")
            fit(model,batch_size,data=data)
            
            output = predict(model,output, test_batch_size, mode = 'fit',label=label)
            #存储每次拟合后的模型
            # save_checkpoint(model, name=model_id,path=checkpoint_path)
        # 只存储最后一次拟合后的模型    
        save_checkpoint(model, name='latest_model',path=checkpoint_path)
        
    # save_checkpoint(model, name='base_model',path=dataset+'/checkpoint/')
    # elif mode == 'check': #check 模式
    #     print("------进入check模式------")
    #     model_id = model.data.predict_time
    #     data = 'valid'
    #     label = ast.literal_eval(label)
    #     # print("------------获取关键实体------------")
    #     # model.data.get_focus_en(label)
    #     # save_checkpoint(model, name="base_model",path=dataset+'/checkpoint/')
    #     output = predict(model,output, test_batch_size, mode = 'prev',label=label)
    #     fit(model,batch_size,data=data)
        
    #     output = predict(model,output, test_batch_size, mode = 'check',label=label)
    #     # save_checkpoint(model, name="latest_model",path=checkpoint_path)
    # # model.data.get_qradua(dataset+'.csv',model.data.predict_time)

    else: #init 模式
        print("------进入init模式------")
        model_id = time.strftime('%Y%m%d%H%M%S', time.localtime())
        metric_history = {}
        loss_history = []
        train_time = []
        evaluate_time = []
        if filter_out:
            monitor = 'filter ' + monitor
        best = 0
        tolerance = early_stop
        for epoch in range(epochs):
            time_start = time.time()
            loss = model.train_epoch(batch_size=batch_size,dataset='train')
            time_end = time.time()
            train_time.append(time_end - time_start)
            loss_history.append(float(loss))
            print('epoch: %d |loss: %f |time: %fs' % (epoch + 1, loss, time_end - time_start))
            if (epoch + 1) % step == 0:
                time_start = time.time()
                metrics = model.test(batch_size=test_batch_size, filter_out=filter_out,tolerance = tolerance)
                time_end = time.time()
                evaluate_time.append(time_end - time_start)

                for key in sorted(metrics.keys()):
                    print(key, ': ', metrics[key], ' |', end='')
                    if key not in metric_history.keys():
                        metric_history[key] = []
                        metric_history[key].append(metrics[key])
                    else:
                        metric_history[key].append(metrics[key])
                print('time: %f' % (time_end - time_start))
                if metric_history[monitor][-1] < metric_history[monitor][best]:
                    # performance decline
                    if early_stop > 0:
                        tolerance -= 1
                        if tolerance <= 0:
                            break
                else:
                    # achieve better performance, save model
                    # save_checkpoint(model, name=model_id,path='./data/temporal/extrapolation/'+dataset+'/checkpoint/')
                    # reset tolerance
                    tolerance = early_stop
                    best = (epoch // step)

        print("\n**********************************finish**********************************\n")
        print("best : ", end='')
        for key in sorted(metric_history.keys()):
            print(key, ' ', metric_history[key][best], ' |', end='')
        print()
        # path = dataset + '/checkpoint/' + model.name + '/' + model_id + '/'
        path = checkpoint_path + model.name + '/' + "base_model" + '/'
        if plot:
            # plot loss and metrics
            from utils.plot import hist_value
            hist_value({'hits@1': metric_history['hits@1'],
                        'hits@3': metric_history['hits@3'],
                        'hits@10': metric_history['hits@10'],
                        'hits@100': metric_history['hits@100']},
                    path=path,
                    metric_name='hits@k',
                    name=model_id + 'valid_hits@k')
            hist_value({'mr': metric_history['mr']},
                    path=path,
                    metric_name='mr',
                    name=model_id + 'valid_mr')
            hist_value({'mrr': metric_history['mrr']},
                    path=path,
                    metric_name='mrr',
                    name=model_id + 'valid_mrr')
            hist_value({'loss': loss_history},
                    path=path,
                    metric_name='loss',
                    name=model_id + 'train_loss')
        # save train history
        data_to_save = metric_history
        data_to_save['loss'] = loss_history
        data_to_save['train_time'] = train_time
        data_to_save['evaluate_time'] = evaluate_time
        save_json(data_to_save, name='train_history', path=path)
        # predict(model, test_batch_size, mode = 'prev',overwrite_flag = True)
        print(path)
        save_checkpoint(model, name='base_model',path=checkpoint_path)
        return
        
        # # put valid_data to train model
        # fit(model,batch_size,data="valid")
        
        # # calculate now_result and compare with prev_result
        # predict(model, test_batch_size, mode = 'now')
        # # save model
        # save_checkpoint(model, name=model.data.predict_time,path=dataset+'/checkpoint/')
    
    #将output 输出成json格式
    output_list = []
    if mode != 'check':
        output = merge_output(event_path,output)

    # 遍历字典并构建列表
    cnt = 0
    for key, value in output.items():
        if key == 'predict':
            output_list.append({"predict": value})
        else:
            minv = min(value['y_data'])
            maxv = max(value['y_data'])
            threshold = model.data.num_relation *0.2 *2
            for i in range(len(value['y_data'])):
                if abs(value['y_data'][i]) < threshold: # 处理小于阈值的数据为概率的方式
                    value['y_data'][i] /= threshold
                elif value['y_data'][i] > 0:
                    value['y_data'][i] /= maxv + threshold
                else:
                    value['y_data'][i] /= abs(minv - threshold)
            entry = {
                "lineGraphName": key,
                "x_data": value['x_data'],
                "y_data": value['y_data']
            }
            output_list.append(entry)
    output_list =sorted_output(output_list)
       
    with open(event_path+'/output.json', 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)

    print(f"数据已成功写入 {event_path}/output.json")# 输出结果
    with open ("result1.txt","a") as f:
        f.write(f"{event_name} : {output_list[-1]['predict']}\n")

def evaluate(model, batch_size, model_id,dataset, data='test', filter_out=False):
    """
    evaluate model in test set or valid set
    :param model_id:
    :param filter_out:
    :param model: model
    :param dataset:
    :param batch_size: batch size
    :param data: dataset
    :return: None
    """
    metrics = model.test(batch_size=batch_size, dataset=data, filter_out=filter_out)
    for key in metrics.keys():
        print(key, ': ', metrics[key], ' |', end='')
    # path = './data/temporal/extrapolation/'+ dataset + '/checkpoint/' + model.name + '/' + model_id + '/'
    path = dataset + '/checkpoint/' + model.name + '/' + model_id + '/' 
    save_json(metrics, name='test_result', path=path)

def predict(model,output, batch_size, mode='prev',label=None,event_time = None):
    """
    Predict event in predict
    :param model: model
    :param batch_size: batch size
    :param mode: mode
    :return: None
    """
    output = model.predict(output,batch_size=batch_size, mode=mode,label=label,event_time = event_time)
    print("predict result has saved\n")
    return output

def read_csv(path_csv):
    data = []
    with open(path_csv,"r",encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if len(row) == 6:
                data.append(row)
    return data

def main(args):
    # set random seed
    set_seed(args.seed)
    # set floating point precision
    set_default_fp(args.fp)
    model_handle = ModelHandle(args.model)
    # set device
    device = set_device(args.gpu)
    # get dataset
    dataset =  read_csv(args.dataset)
    # load checkpoint
    checkpoint_path= './'+args.dir_path+'/'+ args.event_name + '/checkpoint/'
    if args.checkpoint is not None:
        # 拟合模型
        
        model = load_checkpoint(args.checkpoint, model_handle, args, device, path= checkpoint_path)
        label = ast.literal_eval(args.label)
        model.data.get_focus_en(label)
        # save_checkpoint(model, name='base_model',path=checkpoint_path)
        if args.mode == 'check':
            time = dt.datetime.strptime(model.data.predict_time, "%Y-%m-%d").date()
            _,model.data.next_data,_ = model.data.read_data(dataset,time,split_flag=True,mode = 'check')
    else:
        # 初始模型
        # load data
        
        data = DataLoader(dataset=dataset,dataset_file=args.dataset,type=model_handle.get_type(args.model),event_name = args.event_name)
        print("!!!!!!",args.time)
        label = ast.literal_eval(args.label)
        flag = data.load(label,args.time)
        if not flag:
            raise Exception("数据集数据量太少 或 事件不清晰 无法训练模型！")
        data.to(device)
        # base model
        if args.config:
            base_model = model_handle.get_base_model(data)
        else:
            base_model = model_handle.get_default_base_model(data)
        # Optimizer
        opt = get_optimizer(args, base_model)
        # model
        model = model_handle.Model(base_model, data, opt)
        model.to(device)

    test_batch_size = args.batch_size
    if args.test_batch_size is not None:
        test_batch_size = args.test_batch_size

    if args.test:
        # evaluate
        model_id = args.checkpoint
        if args.checkpoint is None:
            raise Exception("You need to load a checkpoint for testing!")
        evaluate(model, test_batch_size,dataset=args.dataset, model_id=model_id, filter_out=args.filter)
    elif args.train:
        # train
        train(model,args.mode,args.epoch, args.batch_size, test_batch_size, args.eva_step, args.early_stop, args.monitor,
              filter_out=args.filter,plot=args.plot,dataset=dataset,event_name=args.event_name,checkpoint = args.checkpoint,label=args.label,device=device,timespan=args.timespan,continue_flag = args.continue_flag,dir_path=args.dir_path,event_time = args.time)
    else :
        # predict
        output = {}
        model_id = args.checkpoint
        label = ast.literal_eval(args.label)
        if args.checkpoint is None:
            raise Exception("You need to load a checkpoint for predicting!")
        predict(model,output, test_batch_size,label)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KGTL')
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="choose model")
    parser.add_argument("--event_name",type=str,required=True,help='event name which used to define model name')
    parser.add_argument("--config", action='store_true', default=False,
                        help="configure parameter")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path and name of model saved')
    # dataset
    parser.add_argument("--dataset",  type=str, help='数据集路径')
    parser.add_argument("--filter", action='store_true', default=False,
                        help="filter triplets. when a query (s,r,?) has multiple objects, filter out others.")
    # Optimizer
    parser.add_argument("--opt", type=str, default='adam',
                        help="optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="weight-decay")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="momentum")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="optimizer parameter")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="optimizer parameter")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="computing running averages of gradient")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="decay factor of the squared gradient")
    parser.add_argument("--amsgrad", action='store_true', default=False,
                        help="Adam optimizer parameter")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # train
    parser.add_argument("--epoch", type=int, default=30,
                        help="learning rate")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size for training.")
    parser.add_argument("--test-batch-size", type=int, default=None,
                        help="batch size for test.")
    parser.add_argument("--eva-step", type=int, default=1,
                        help="evaluate model on valid set after 'eva-step' step of training.")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="patience for early stop.")
    parser.add_argument("--monitor", type=str, default='mrr',
                        help="monitor metric for early stop ")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="plot loss and metrics.")
    # test
    parser.add_argument("--test", action='store_true', default=False,
                        help="evaluate model on test set, and notice that you must load a checkpoint for this.")
    # train
    parser.add_argument("--train", action='store_true', default=False,
                        help="train the model.")
    # predict
    parser.add_argument("--predict", action='store_true', default=False,
                        help="evaluate model on predict set, and notice that you must load a checkpoint for this.") 
    # fit
    parser.add_argument("--fit", action='store_true', default=False,
                        help="fit model on test set, and notice that you must load a checkpoint for this.") 
    # other
    parser.add_argument("--fp", type=str, default='fp32',
                        help="Floating point precision (fp16, bf16, fp32 or fp64) ")
    parser.add_argument("--gpu", type=int, default=-2,
                        help="Use the GPU with the lowest memory footprint by default. "
                             "Specify a GPU by setting this parameter to a GPU id which equal to or greater than 0."
                             "Set this parameter to -1 to use the CPU."
                        )
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed.")
    parser.add_argument('--label', type=str, help='嵌套列表，如 "[[], [], []]"')
    parser.add_argument('--time',type=str,default=None,help='事件发生时间') 
    parser.add_argument('--timespan',type=int,help='预测时间间隔')
    parser.add_argument('--continue_flag',action='store_true',default=False,help='是否是继续init后的fit操作') 
    parser.add_argument('--mode',type=str,default='init',help='模式选择 init or fit or check')
    parser.add_argument('--dir_path',type=str,default='temporal_data',help='数据存放路径')


    args_parsed = parser.parse_args()

    main(args_parsed)
