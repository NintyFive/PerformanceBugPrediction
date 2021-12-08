#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.utils import resample
import os

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import ParameterGrid
import json
from imblearn.over_sampling import SMOTE
import time

algo_dict = {}
algo_dict['CNB'] = ComplementNB
algo_dict['LR'] = LogisticRegression
algo_dict['DT'] = DecisionTreeClassifier
algo_dict['MLP'] = MLPClassifier
algo_dict['SVM'] = LinearSVC
algo_dict['XGBoost'] = xgb.XGBClassifier
algo_dict['RF'] = RandomForestClassifier

algo_param_grid = {}

algo_param_grid['MLP'] = {
    'hidden_layer_sizes': [(64, 32), (32, 16), (16, 8), (8,)],
    'activation': ['tanh', 'relu', 'identity', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [1e-3, 1e-4, 5e-4, 5e-3, 1e-2],
    'max_iter': [100, 300]
}

algo_param_grid['LR'] = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 400, 500],
    'class_weight': ['balanced']
}

algo_param_grid['DT'] = {
    'max_depth': [6, 8, 10, 12, 16],
    'max_features': ['auto', 'log2', 'sqrt'],
    'class_weight': ['balanced']
}

algo_param_grid['RF'] = {'n_estimators':[100, 200, 300, 400, 500], 
                         'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5], 
                         'max_features':[0.1, 0.2, 0.3, 0.4, 0.5],
                         'max_depth': [6, 8, 12, 16, 20],
                         'n_jobs': [2], 
                         'criterion':['entropy'],
                         'min_samples_leaf':[1, 2, 4, 8],
                         'min_samples_split':[2, 4, 8, 16],  
                         'class_weight':['balanced_subsample'],
                         'verbose':[0] }
 
algo_param_grid['XGBoost'] = {'n_estimators':[100, 200, 300, 400, 500],
                         'eta': [1e-3, 1e-4, 5e-4, 5e-3, 1e-2],
                         'max_depth': [6, 8, 12, 16, 20],
                         'colsample_bytree':[0.1, 0.2, 0.3, 0.4, 0.5],
                         'subsample':[0.1, 0.2, 0.3, 0.4, 0.5],
                         'nthread':[4],
                         'objective':['binary:logistic'], 
                         'verbosity':[0] }

algo_param_grid['CNB'] = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1]}

algo_param_grid['SVM'] = {#'penalty': ['l1', 'l2'],
                          'tol': [1e-4, 5e-4, 5e-3, 1e-3, 1e-2],
                          'C': [1e-2, 1e-1, 1],
                          'loss': ['hinge', 'squared_hinge'],
                          'loss': ['squared_hinge'],
                          'dual': [False, True],
                          'class_weight': ['balanced']
                         }

# Number of neighbors used in SMOTE technique
for algo in algo_dict.keys():
    algo_param_grid[algo]['k_neighbors'] = [3, 5, 7, 11, 17]

def evaluate(dataset, train_index, test_index, algo, param):
    X = dataset.drop(columns=['File', 'label'], axis=1)
    y = dataset['label']
    
    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    
    k_neighbors = param['k_neighbors']
    oversample = SMOTE(k_neighbors = k_neighbors)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    param.pop('k_neighbors', None)
    model = algo(**param)
    
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        prob_pos = probs[:,1]
    else:  # use decision function
        prob_pos = model.decision_function(X_test)
        prob_pos =  (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob_pos, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    precision, recall, _ = metrics.precision_recall_curve(y_test, prob_pos)
    pr_auc = metrics.auc(recall, precision)
    
    max_mcc = -1
    optimal_threshold = 0
    for threshold in thresholds:
        y_pred = (prob_pos >= threshold).astype(bool) 
        mcc = metrics.matthews_corrcoef(y_test, y_pred)
        if mcc > max_mcc:
            max_mcc = mcc
            optimal_threshold = threshold


    
    result = {}
    result['auc'] = auc
    result['mcc'] = max_mcc
    result['pr_auc'] = pr_auc
    
    #restore param for SMOTE technique
    param['k_neighbors'] = k_neighbors 
    return result

def parameter_optimize(project_data_dir, dataset, algo, metrics_name=''):
    start_time = time.time()
    N = 100
    train_list = []
    test_list = []
    for i in range(N):
        train_index = resample(dataset.index.values, random_state=i, stratify=dataset['label'])
        test_index = list(set(dataset.index.values) - set(train_index))
        train_list.append(train_index)
        test_list.append(test_index)
        
        
    results_dict = {}
    test_results_dict = {}
    print(algo)
    params = list(ParameterGrid(algo_param_grid[algo]))
    print('len params: ', len(params))
    results_dict[algo] = [[] for i in range(len(params))]
    for i in range(N):
        
        train_index = resample(train_list[i], replace=False, n_samples = int(len(train_list[i])*0.7))
        test_index = list(set(train_list[i]) - set(train_index))
        for index, param in enumerate(params):
            try:
                result = evaluate(dataset, train_index, test_index, algo_dict[algo], param)
                results_dict[algo][index].append(result)
            except Exception as e:
                continue


            
    max_pr_auc = 0
    max_param = 0
    for i in range(len(params)):
        sum_pr_auc = 0
        for index in range(N):
            sum_pr_auc += results_dict[algo][i][index]['pr_auc']
        if sum_pr_auc > max_pr_auc:
            max_pr_auc = sum_pr_auc
            max_param = i
            
            
    with open(os.path.join(project_data_dir, 'best_params%s' % metrics_name, '%s.txt' % algo), 'w') as file:
        file.write(str(max_param))
        
    test_results_dict[algo] = []
    for i in range(N):
        print("Test iteration: ", i)
        train_index = train_list[i]
        test_index = test_list[i]
        result = evaluate(dataset, train_index, test_index, algo_dict[algo], params[max_param])
        test_results_dict[algo].append(result)
    
    with open(os.path.join(project_data_dir, 'best_params%s' % metrics_name, '%s_test_results.json' % algo), 'w') as file:
        file.write(json.dumps(test_results_dict))

    print("Time spent in parameter tunning --- %s seconds ---" % (time.time() - start_time))
