import csv
import os
import pickle
from pathlib import Path
from ast import literal_eval

import pandas as pd
import numpy as np
import random

from geopy.distance import geodesic
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_implicit_ml_1m_dataset(file, file_name, embed_dim=8, maxlen=40, neg_sample=100):
    """
    :param file_name: 暂存数据点的文件夹名
    :param neg_sample: 负样本的个数
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :return: user_num, item_num, train_df, test_df
    """
    # create datasetz
    def df_to_list(data):
        # 将序列填充为长为40(maxlen)的序列
        return [data['user_id'].values, pad_sequences(data['hist'], maxlen=maxlen),
                data['pos_item'].values, data['neg_item'].values,
                data['distance_pos'].apply(get_mean).values,
                data['distance_neg'].apply(get_mean).values]

    def distance_limit(x):
        if x < 1:
            return 1
        return x

    def get_mean(d_list):
        # d_list = literal_eval(d_list)
        d_list = d_list[-maxlen:]
        d_sum = 0
        for d in d_list:
            d_sum += d
        return d_sum/len(d_list)

    file_name = file_name
    if not os.path.exists('data/'+file_name):
        os.makedirs('data/'+file_name)
    print('==========Data Preprocess Start============')
    random.seed(10)
    train_file = Path('data/'+file_name+'/train.csv')
    test_file = Path('data/'+file_name+'/test.csv')
    new_test_file = Path('data/' + file_name + '/new_test.csv')
    if train_file.is_file() and test_file.is_file():
        train = pd.read_csv('data/'+file_name+'/train.csv')
        test = pd.read_csv('data/'+file_name+'/test.csv')
        new_test = pd.read_csv('data/' + file_name + '/new_test.csv')
        train['hist'] = train['hist'].apply(literal_eval)
        test['hist'] = test['hist'].apply(literal_eval)
        new_test['hist'] = new_test['hist'].apply(literal_eval)
        train['distance_pos'] = train['distance_pos'].apply(literal_eval)
        test['distance_pos'] = test['distance_pos'].apply(literal_eval)
        new_test['distance_pos'] = new_test['distance_pos'].apply(literal_eval)
        train['distance_neg'] = train['distance_neg'].apply(literal_eval)
        test['distance_neg'] = test['distance_neg'].apply(literal_eval)
        new_test['distance_neg'] = new_test['distance_neg'].apply(literal_eval)
        train_X = df_to_list(train)
        test_X = df_to_list(test)
        new_test_X = df_to_list(new_test)
        feature_columns = pickle.load(open('data/'+file_name+'/feature_columns.txt', 'rb'))
        print('============Data Preprocess End=============')
        return feature_columns, train_X, test_X,new_test_X
    names = ['user_id', 'item_id', 'label', 'Timestamp', 'lat', 'lon']
    data_df = pd.read_csv(file, header=None, sep='\t', names=names)

    # 包含经纬度的map
    p = data_df.groupby('item_id')[['lat', 'lon']].mean()
    lat_lon_map = p.to_dict()

    train_data, test_data, new_test_data = [], [], []
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
                return neg

        def get_distance(seq, pos):
            distance = 0
            lat1, lon1 = lat_lon_map['lat'][seq], lat_lon_map['lon'][seq]
            lat2, lon2 = lat_lon_map['lat'][pos], lat_lon_map['lon'][pos]
            distance += geodesic((lat1, lon1), (lat2, lon2)).km
            if distance < 1:
                distance = 1
            return distance

        def get_distance_list(seq, pos):
            distance_list = []
            for s in seq:
                distance_list.append(get_distance(s,pos))
            return distance_list

        neg_list = [gen_neg() for i in range(len(pos_list) + neg_sample)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if len(hist_i) > 10:
                hist_i = hist_i[-10:]
            if i == len(pos_list) - 1:
                for neg in neg_list[i:]:
                    test_data.append([user_id, hist_i, pos_list[i], neg,
                                      get_distance_list(hist_i, pos_list[i]),
                                      get_distance_list(hist_i, neg)])
                if pos_list[i] not in hist_i:
                    for neg in neg_list[i:]:
                        new_test_data.append([user_id, hist_i, pos_list[i], neg,
                                              get_distance_list(hist_i, pos_list[i]),
                                              get_distance_list(hist_i, neg)])
            else:
                train_data.append([user_id, hist_i, pos_list[i], neg_list[i],
                                   get_distance_list(hist_i, pos_list[i]),
                                   get_distance_list(hist_i, neg_list[i])])

    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim)]
    pickle.dump(feature_columns, open('data/'+file_name+'/feature_columns.txt', 'wb'))
    # shuffle 将序列的所有元素随机排序。
    random.shuffle(train_data)
    random.shuffle(test_data)
    print('==================Saving===================')
    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'pos_item', 'neg_item',
                                              'distance_pos', 'distance_neg'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'pos_item', 'neg_item',
                                            'distance_pos', 'distance_neg'])
    new_test = pd.DataFrame(new_test_data, columns=['user_id', 'hist', 'pos_item', 'neg_item',
                                            'distance_pos', 'distance_neg'])
    # train['distance_pos'] = train['distance_pos'].apply(lambda x: distance_limit(x))
    # train['distance_neg'] = train['distance_neg'].apply(lambda x: distance_limit(x))
    # test['distance_pos'] = test['distance_pos'].apply(lambda x: distance_limit(x))
    # test['distance_neg'] = test['distance_neg'].apply(lambda x: distance_limit(x))
    train.to_csv('data/'+file_name+'/train.csv', index=False)
    test.to_csv('data/'+file_name+'/test.csv', index=False)
    new_test.to_csv('data/'+file_name+'/new_test.csv', index=False)
    print('==================Padding===================')
    train_X = df_to_list(train)
    test_X = df_to_list(test)
    new_test_X = df_to_list(new_test)
    print('============Data Preprocess End=============')
    return feature_columns, train_X, test_X, new_test_X
