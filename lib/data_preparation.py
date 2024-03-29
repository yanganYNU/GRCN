# -*- coding:utf-8 -*-
"""
Created on Thu Apr 25 15:12:27 2021

@author: yang an
"""

import numpy as np
# import mxnet as mx

from .utils import get_sample_indices


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0,2,1,3)
    val = (val).transpose(0,2,1,3)
    test =(test).transpose(0,2,1,3)

    return {'mean': mean, 'std': std}, train, val, test


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data
    merge: boolean, default False,
           whether to merge training set and validation set to train model
    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']
    all_samples = []
    for idx in range(data_seq.shape[0]):
        # smaple自动定义成一个元组接受输出结果
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        # python的元组拆包，sample为一个元组，拆包的要求就是需要拆的数据的个数要与接受数据的变量的个数相同
        week_sample, day_sample, hour_sample, target = sample

        # np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))
        # np.expand_dims(week_sample, axis=0) 增加一个维度，此前维度为（0，1，2）分别为1、2、3维度，现在为（0，1，2，3）
        # .transpose((0, 2, 3, 1))更换坐标轴，原本为（0，1，2，3）现在改为（0，2，3，1）
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))

    # 样本分割线，0.6的分割比例
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        # 训练集，split_line1为60%的分割线，也就是将数据集的0-60%部分为训练集
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        # merge=True ，将20%的验证集也合并一起作为训练集
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    # 验证集，split_line2为80%的分割线，也就是将数据集的60-80%部分为验证集
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]

    # 测试集，split_line2为80%的分割线，也就是将数据集的80-100%部分为测试集
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    # 同前面一样，training_set、validation_set、testing_set自动成为一个包含四个元组的元组
    # 自动拆包
    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    # 分别打印出（训练集、验证集、测试集）的（周、日、邻近）的数据维度
    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    # 对（训练集、验证集、测试集）的（周、日、邻近）的做归一化处理
    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week, val_week, test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day, val_day, test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour, val_hour, test_hour)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data
