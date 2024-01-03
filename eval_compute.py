# -*- coding: utf-8 -*-
import json
import os
import math
import copy
import argparse
import pandas as pd

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Example usage of argparse module')
# 添加命令行参数
parser.add_argument('--model_name', type=str, default='llama')
parser.add_argument('--model_version', type=str)
parser.add_argument('--bench_version', type=str, default='test_data_v1.1')
parser.add_argument('--save_excel_path', type=str, default='./result.xlsx')

# 解析命令行参数
args = parser.parse_args()
def metric_compute(model_name=args.model_name, model_version=args.model_version, bench_version=args.bench_version, save_path=args.save_excel_path):
    folder = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/blue_whale_data/eval_output/{}/{}/{}'.format(model_name,
                                                                                                                           model_version, bench_version)
    
    meta = {
        "C-eval":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.49,
            "source":"general"
        },
        "LogiQA2.0中文":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.44,
            "source":"general"
        },
        "LogiQA2.0英文":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.49,
            "source":"general"
        },
        "WTQ-short":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.4395,
            "source":"general"
        },
        "agieval-高考历史选择":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.53,
            "source":"general"
        },
        "agieval-高考数学填空（markdown公式理解）":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.00001,
            "source":"general"
        },
        "agieval-高考英语选择":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.74,
            "source":"general"
        },
        "agieval-高考语文选择":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.31,
            "source":"general"
        },
        "csiper":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.5038,
            "source":"general"
        },
        "dureader":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.3085,
            "source":"general"
        },
        "dusql":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.5311,
            "source":"general"
        },
        "nl2sql_tabular":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.3374,
            "source":"general"
        },
        "wikisql":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.6109,
            "source":"general"
        },
        "meeting_merge_summary":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.6401,
            "source":"general"
        },
        "meeting_split_summary":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.4085,
            "source":"general"
        },
        "PDC虚拟号_商家营业状态异常原因":{
            "metric":"macro_prf.f1",
            "chatGPT_metric_value":0.4672,
            "source":"meituan"
        },
        "买菜虚拟号_骑手提前送达":{
            "metric":"binary_prf.f1",
            "chatGPT_metric_value":0.4038,
            "source":"meituan"
        },
        "到综求合作":{
            "metric":"macro_prf.f1",
            "chatGPT_metric_value":0.3844,
            "source":"meituan"
        },
        "到综虚拟号_地址不一致":{
            "metric":"binary_prf.f1",
            "chatGPT_metric_value":0.5217,
            "source":"meituan"
        },
        "危+急风险会话识别":{
            "metric":"binary_prf.recall",
            "chatGPT_metric_value":0.4796,
            "source":"meituan"
        },
        "在线意图_住宿":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.42,
            "source":"meituan"
        },
        "外卖虚拟号_上报异常":{
            "metric":"binary_prf.f1",
            "chatGPT_metric_value":0.695,
            "source":"meituan"
        },
        "学城文档问答":{
            "metric":"rouge.rouge",
            "chatGPT_metric_value":0.7295,
            "source":"meituan"
        },
        "智能断线":{
            "metric":"binary_prf.precision",
            "chatGPT_metric_value":0.2432,
            "source":"meituan"
        },
        "断线外呼-住宿":{
            "metric":"acc.accuracy",
            "source":"meituan",
            "chatGPT_metric_value":0.39,
        },
        "组件点选到家质检":{
            "metric":"acc.accuracy",
            "chatGPT_metric_value":0.4583,
            "source":"meituan"
        },
        "通用检测项_涉黄涉暴涉政":{
            "metric":"binary_prf.f1",
            "chatGPT_metric_value":0.3125,
            "source":"meituan"
        },
        "酒店虚拟号_满房超售":{
            "metric":"binary_prf.f1",
            "chatGPT_metric_value":0.8193,
            "source":"meituan"
        },
        "阿波罗质检_探寻合作意向结果":{
            "metric":"macro_prf.f1",
            "chatGPT_metric_value":0.2313,
            "source":"meituan"
        },
        "阿波罗质检_邀约上门结果":{
            "metric":"macro_prf.f1",
            "chatGPT_metric_value":0.2107,
            "source":"meituan"
        },
        "智能销售_挖需-了解商户意图":{
            "metric":"binary_prf.f1",
            "source":"meituan",
            "chatGPT_metric_value":0.4545,
        },
        "智能销售_推方案-价格或方案搭配优化":{
            "metric":"binary_prf.f1",
            "source":"meituan",
            "chatGPT_metric_value":0.6087,
        },
        '智能销售_破冰-上次沟通和问题处理':{
            "metric":"binary_prf.f1",
            "source":"meituan",
            "chatGPT_metric_value":0.1333,
        },
        '智能销售_破冰-节日活动':{
            "metric":"binary_prf.f1",
            "source":"meituan",
            "chatGPT_metric_value":0.4545,
        },
        '智能销售_解异议-无效果_产品价值':{
            "metric":"binary_prf.f1",
            "source":"meituan",
            "chatGPT_metric_value":0.7789,
        }
    }

    general_number = 0
    meituan_number = 0
    all_metric_value = 0
    all_metric_value_chatGPT = 0
    all_metric_win_rate = 0
    all_metric_gap10_rate = 0
    all_metric_gap30_rate = 0
    general_dataset = list()
    general_dataset_number = 0
    general_metric_accumulation = 0
    general_metric_accumulation_chatGPT = 0
    general_metric_value = 0
    general_metric_value_chatGPT = 0
    general_metric_win_number = 0
    general_metric_win_rate = 0
    general_metric_gap10_number = 0
    general_metric_gap10_rate = 0
    general_metric_gap30_number = 0
    general_metric_gap30_rate = 0
    meituan_dataset = list()
    meituan_dataset_number = 0
    meituan_metric_accumulation = 0
    meituan_metric_accumulation_chatGPT = 0
    meituan_metric_value = 0
    meituan_metric_value_chatGPT = 0
    meituan_metric_win_number = 0
    meituan_metric_win_rate = 0
    meituan_metric_gap10_number = 0
    meituan_metric_gap10_rate = 0
    meituan_metric_gap30_number = 0
    meituan_metric_gap30_rate = 0
    eval_results_missing = list()
    additional_info = list()
    df = {}
    for i, file in enumerate(meta.keys()):
        if meta[file]['source']=='general':
            general_number+=1
        else:
            meituan_number+=1
        try:
            with open(os.path.join(folder,file,'summarized_test_metrics.json'))as fr:
                js = json.load(fr)[0]
                metric_name,metric_value = meta[file]['metric'].split('.')
                metric = js[metric_name][metric_value]
                if metric_name != 'rouge':
                    metric/=100
                meta[file]['model_exp']=metric
                
                if meta[file]['source']=='general':
                    general_dataset.append(file)
                    general_dataset_number+=1
                    general_metric_accumulation+=metric
                    general_metric_accumulation_chatGPT+=meta[file]['chatGPT_metric_value']
                    if metric>meta[file]['chatGPT_metric_value']:
                        general_metric_win_number+=1
                    if metric/meta[file]['chatGPT_metric_value']>=0.7:
                        general_metric_gap30_number+=1
                    if metric/meta[file]['chatGPT_metric_value']>=0.9:
                        general_metric_gap10_number+=1
                else:
                    meituan_dataset.append(file)
                    meituan_dataset_number+=1
                    meituan_metric_accumulation+=metric
                    meituan_metric_accumulation_chatGPT+=meta[file]['chatGPT_metric_value']
                    if metric>meta[file]['chatGPT_metric_value']:
                        meituan_metric_win_number+=1
                    if metric/meta[file]['chatGPT_metric_value']>=0.7:
                        meituan_metric_gap30_number+=1
                    if metric/meta[file]['chatGPT_metric_value']>=0.9:
                        meituan_metric_gap10_number+=1
                df[i] = []
                df[i].append(file)
                df[i].append(metric)
        except:
            eval_results_missing.append(file)
            continue
    pd.DataFrame.from_dict(df).to_excel(save_path)
    print("评测结果已存至：{}\n".format(save_path))
    if general_dataset_number>0:
        general_metric_value = general_metric_accumulation/general_dataset_number
        general_metric_win_rate = general_metric_win_number/general_dataset_number
        general_metric_gap10_rate = general_metric_gap10_number/general_dataset_number
        general_metric_gap30_rate = general_metric_gap30_number/general_dataset_number
        general_metric_value_chatGPT = general_metric_accumulation_chatGPT/general_dataset_number
    if meituan_dataset_number>0:
        meituan_metric_value = meituan_metric_accumulation/meituan_dataset_number
        meituan_metric_win_rate = meituan_metric_win_number/meituan_dataset_number
        meituan_metric_gap10_rate = meituan_metric_gap10_number/meituan_dataset_number
        meituan_metric_gap30_rate = meituan_metric_gap30_number/meituan_dataset_number
        meituan_metric_value_chatGPT = meituan_metric_accumulation_chatGPT/meituan_dataset_number
    if general_dataset_number+meituan_dataset_number>0:
        all_metric_value = (general_metric_accumulation+meituan_metric_accumulation)/(general_dataset_number+meituan_dataset_number)
        all_metric_win_rate = (general_metric_win_number+meituan_metric_win_number)/(general_dataset_number+meituan_dataset_number)
        all_metric_gap10_rate = (general_metric_gap10_number+meituan_metric_gap10_number)/(general_dataset_number+meituan_dataset_number)
        all_metric_gap30_rate = (general_metric_gap30_number+meituan_metric_gap30_number)/(general_dataset_number+meituan_dataset_number)
        all_metric_value_chatGPT = (general_metric_accumulation_chatGPT+meituan_metric_accumulation_chatGPT)/(general_dataset_number+meituan_dataset_number)
    print("-----自建榜单-汇总-----")
    if all_metric_value_chatGPT>0: 
        print("平均得分={0:.4f}".format(all_metric_value))
        print("平均胜率={0:.4f}".format(all_metric_win_rate))
        print("能力达成率={0:.4f}, {1:.4f}/{2:.4f}".format(all_metric_value/all_metric_value_chatGPT,all_metric_value,all_metric_value_chatGPT))
        print("能力达成chatGPT90%以上任务占比={0:.4f}".format(all_metric_gap10_rate))
        print("能力达成chatGPT70%以上任务占比={0:.4f}".format(all_metric_gap30_rate))
        print("评测数据集数量={}/{}".format(general_dataset_number+meituan_dataset_number,general_number+meituan_number))
        print()
    else:
        print("暂无评测结果\n")
    print("-----自建榜单-公开-----")
    if general_metric_value_chatGPT>0:
        print("平均得分={0:.4f}".format(general_metric_value))
        print("平均胜率={0:.4f}".format(general_metric_win_rate))
        print("能力达成率={0:.4f}, {1:.4f}/{2:.4f}".format(general_metric_value/general_metric_value_chatGPT,general_metric_value,general_metric_value_chatGPT))
        print("能力达成chatGPT90%以上任务占比={0:.4f}".format(general_metric_gap10_rate))
        print("能力达成chatGPT70%以上任务占比={0:.4f}".format(general_metric_gap30_rate))
        print("评测数据集数量={}/{}".format(general_dataset_number,general_number))
        print()
    else:
        print("暂无评测结果\n")
    print("-----自建榜单-业务-----")
    if meituan_metric_value_chatGPT>0:
        print("平均得分={0:.4f}".format(meituan_metric_value))
        print("平均胜率={0:.4f}".format(meituan_metric_win_rate))
        print("能力达成率={0:.4f}, {1:.4f}/{2:.4f}".format(meituan_metric_value/meituan_metric_value_chatGPT,meituan_metric_value,meituan_metric_value_chatGPT))
        print("能力达成chatGPT90%以上任务占比={0:.4f}".format(meituan_metric_gap10_rate))
        print("能力达成chatGPT70%以上任务占比={0:.4f}".format(meituan_metric_gap30_rate))
        print("评测数据集数量={}/{}".format(meituan_dataset_number,meituan_number))
        print()
    else:
        print("暂无评测结果\n")
    print("-----评测详情-----")
    for file in meta.keys():
        if 'model_exp' in meta[file]:
            additional_info.append(file+' source:{0}\n基线组:{1:4f} 实验组:{2:.4f} 能力达成率:{3:4f}'.format(meta[file]['source'],meta[file]['chatGPT_metric_value'],meta[file]['model_exp'],meta[file]['model_exp']/(meta[file]['chatGPT_metric_value']+1e-5)))
    if len(additional_info)>0:
        print('\n----------\n'.join(additional_info))
    else:
        print("暂无评测结果\n")
    if len(eval_results_missing)>0:
        print("-----缺失评测结果的数据集-----")
        print('\n'.join(eval_results_missing))
metric_compute()