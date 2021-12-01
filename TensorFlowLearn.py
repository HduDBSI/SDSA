import sys
import os
from tqdm import tqdm
import pandas as pd
import argparse
from model.customLayer import LinearModel
import tensorflow as tf
import numpy as np
from geopy.distance import geodesic
from utils import *
from evaluate import *
from time import time
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--len_Seq', type=int, default=3)
    parser.add_argument('--len_Tag', type=int, default=1)
    parser.add_argument('--len_Pred', type=int, default=1)
    parser.add_argument('--neg_sample', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--l2_lambda', type=float, default=1e-6)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--mode', type=str, default='inner')
    parser.add_argument('--mode2', type=int, default='2')
    parser.add_argument('--data_path', type=str,
                        default='/home/wf/shenyi/Data/Gowalla/output/Gowalla_ML.data')
                        # default='/home/wf/shenyi/Data/dataset_tsmc2014/output/NYC_ML.data')
                        # default='/home/wf/shenyi/Data/dataset_tsmc2014/output/TKY_ML.data')
    parser.add_argument('--data_name', type=str, default='Gowalla')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Get Params
    args = parse_args()
    len_Seq = args.len_Seq  # 序列的长度
    len_Tag = args.len_Tag  # 训练时目标的长度
    len_Pred = args.len_Pred  # 预测时目标的长度
    batch_size = args.batch_size  # 把数据分为几个部分
    emb_size = args.emb_size
    neg_sample = args.neg_sample
    K = [1, 3, 5, 10]
    l2_lambda = args.l2_lambda
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    mode = args.mode
    mode2 = str(args.mode2)
    emb_reg = 1e-6

    # make datasets
    print('==> make datasets <==')
    file = args.data_path
    file_name = args.data_name
    names = ['user', 'item', 'rateing', 'timestamps', 'lat', 'lon']
    # data = pd.read_csv(file, header=None, sep='\t', names=names)
    feature_columns, train, test, new_test = create_implicit_ml_1m_dataset(file, file_name, emb_size, len_Seq,
                                                                           neg_sample)
    # build model
    model = LinearModel(emb_size, feature_columns, len_Seq, emb_reg, mode, mode2)
    model.summary()
    # model compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    results = []
    new_results = []
    # 第一列代表epoch
    results.append([])
    new_results.append([])
    results_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_results_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(K)):
        # 每一个K有两列,分别为HR和NDCG
        results.append([])
        results.append([])
        new_results.append([])
        new_results.append([])
    count = 0

    for epoch in range(1, num_epochs + 1):
        # train
        t1 = time()
        model.fit(
            train,
            None,
            epochs=1,
            batch_size=batch_size,
        )
        # test
        t2 = time()
        if epoch % 5 == 0:
            results[0].append(epoch)
            new_results[0].append(epoch)
            print('=================Next recommendation===================')
            for i in range(len(K)):
                hit_rate, ndcg, _ = evaluate_model(model, test, K[i])
                print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, TOP@%d'
                      % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, K[i]))
                results[2 * i + 1].append(hit_rate)
                results[(2 * i + 2)].append(ndcg)
            # 如果最后一行的hr@10最大则更新
            if results[7][-1] > results_max[7]:
                for i in range(len(results)):
                    results_max[i] = results[i][-1]
                count = 0
            else:
                count = count + 1
            print('=================Next New recommendation===================')
            for i in range(len(K)):
                new_hit_rate, new_ndcg, _ = evaluate_model(model, new_test, K[i])
                print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, TOP@%d'
                      % (epoch, t2 - t1, time() - t2, new_hit_rate, new_ndcg, K[i]))
                new_results[2 * i + 1].append(new_hit_rate)
                new_results[(2 * i + 2)].append(new_ndcg)
                # 如果最后一行的hr@10最大则更新
            if new_results[7][-1] > new_results_max[7]:
                for i in range(len(new_results)):
                    new_results_max[i] = new_results[i][-1]
            if count >= 10:
                break


            def write_log(r, rm, is_new=''):
                # write log
                df = pd.DataFrame(r).T
                df.columns = ['Iteration', 'HR@1', 'NDCG@1',
                              'HR@3', 'NDCG@3',
                              'HR@5', 'NDCG@5',
                              'HR@10', 'NDCG@10']
                df.to_csv('log/DAttRec_{}_log_d_{}_maxlen_{}_dim_{}.csv'.format(
                    is_new, file_name, len_Seq, emb_size), index=False)
                with open('log/DAttRec_{}_maxLog_d_{}_maxlen_{}_dim_{}.csv'.format(
                        is_new, file_name, len_Seq, emb_size), 'w', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(rm)


            write_log(results, results_max)
            # 表示记录next new
            write_log(new_results, new_results_max, is_new='new')
