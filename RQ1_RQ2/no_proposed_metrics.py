#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.utils import resample
import pandas as pd
import numpy as np
import os
import math
import collections
import shutil

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from multiprocessing import Process, Manager

from sklearn.model_selection import ParameterGrid
import json
from imblearn.over_sampling import SMOTE
from evaluate_functions import *

if __name__=='__main__':
    proposed_features = ['num_if_inloop', 'num_loop_inif', 'num_nested_loop', 'num_nested_loop_incrit',
                  'synchronization', 'thread', 'io_in_loop', 'database_in_loop', 'collection_in_loop',
                  'io', 'database', 'collection', 'recursive']

    data_dir = 'data'

    project_list = os.listdir(data_dir)

    for project in project_list:
        print('project', project)
        project_data_dir = os.path.join(data_dir, project)
        if not os.path.exists(project_data_dir):
            os.makedirs(project_data_dir)
    
        dataset = pd.read_csv(os.path.join(project_data_dir, 'dataset_labeled.csv'))
        dataset = dataset.rename(columns={'CountLineCode': 'LOC', 'CountLineComment': 'CL', 'CountStmt': 'nstmt', 'MaxNesting': 'MNL', 'RatioCommentToCode': 'RCC', 'Cyclomatic': 'CC', 'CountInput': 'FANIN', 'CountOutput': 'FANOUT'})

        print('Number of total amount of methods: ', dataset.shape[0])
        print('Number of buggy files: ', dataset[dataset['label']==1].shape[0])

        metrics_name = '_no_proposed_metrics'
        metrics_list = proposed_features
        if not os.path.exists(os.path.join(project_data_dir, 'best_params%s' % metrics_name)):
            os.mkdir(os.path.join(project_data_dir, 'best_params%s' % metrics_name))
        print('Start evaluate model performance with', ' '.join(metrics_name.split('_')[1:]))
        
        remain_dataset = dataset.drop(labels=metrics_list, axis=1)
        print('Shape: ', remain_dataset.shape)
        process = []
        for algo in algo_dict.keys():
            p = Process(target=parameter_optimize, args=(project_data_dir, remain_dataset, algo, metrics_name, ))
            process.append(p)
            p.start()
        for process_index, p in enumerate(process):
            print(p)
            p.join()

