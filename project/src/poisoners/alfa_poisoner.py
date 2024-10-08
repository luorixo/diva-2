import argparse
import glob
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
# from sklearn.utils.fixes import loguniform
from scipy.stats import loguniform


from utils.alfa import alfa
from utils.utils import (create_dir, open_csv, open_json, to_csv,
                                      to_json, transform_label)

# Ignore warnings from optimization.
warnings.filterwarnings('ignore')

ALFA_MAX_ITER = 5  # Number of iterations for ALFA.
N_ITER_SEARCH = 50  # Number of iterations for SVM parameter tuning.
SVM_PARAM_DICT = {
    'C': loguniform(0.01, 10),
    'gamma': loguniform(0.01, 10),
    'kernel': ['rbf'],
}

def get_y_flip(X_train, y_train, rate, svc):
    if rate == 0:
        return y_train

    # Transform labels from {0, 1} to {-1, 1}
    y_train = transform_label(y_train, target=-1)
    y_flip = alfa(X_train, y_train,
                  rate,
                  svc_params=svc.get_params(),
                  max_iter=ALFA_MAX_ITER)
    # Transform label back to {0, 1}
    y_flip = transform_label(y_flip, target=0)
    return y_flip

def compute_and_save_flipped_data(X_train, y_train, X_test, y_test, clf, path_output_base, cols, advx_range):
    acc_train_clean = clf.score(X_train, y_train)
    acc_test_clean = clf.score(X_test, y_test)

    accuracy_train_clean = [acc_train_clean] * len(advx_range)
    accuracy_test_clean = [acc_test_clean] * len(advx_range)
    accuracy_train_poison = []
    accuracy_test_poison = []
    path_poison_data_list = []

    for rate in advx_range:
        path_poison_data = '{}_alfa_svm_{:.2f}.csv'.format(path_output_base, np.round(rate, 2))
        try:
            if os.path.exists(path_poison_data):
                X_train, y_flip, _ = open_csv(path_poison_data)
            else:
                time_start = time.time()
                y_flip = get_y_flip(X_train, y_train, rate, clf)
                time_elapse = time.time() - time_start
                print('Generating {:.0f}% poison labels took {:.1f}s'.format(rate * 100, time_elapse))
                to_csv(X_train, y_flip, cols, path_poison_data)
            svm_params = clf.get_params()
            clf_poison = SVC(**svm_params)
            clf_poison.fit(X_train, y_flip)
            acc_train_poison = clf_poison.score(X_train, y_flip)
            acc_test_poison = clf_poison.score(X_test, y_test)
        except Exception as e:
            print(e)
            acc_train_poison = 0
            acc_test_poison = 0
        print('P-Rate [{:.2f}] Acc  P-train: {:.2f} C-test: {:.2f}'.format(rate * 100, acc_train_poison * 100, acc_test_poison * 100))
        path_poison_data_list.append(path_poison_data)
        accuracy_train_poison.append(acc_train_poison)
        accuracy_test_poison.append(acc_test_poison)
    return (accuracy_train_clean,
            accuracy_test_clean,
            accuracy_train_poison,
            accuracy_test_poison,
            path_poison_data_list)

def alfa_poison(dataset, advx_range, path_output):
    train_path = dataset['train']
    test_path = dataset['test']

    create_dir(os.path.join(path_output, 'alfa_svm'))

    dataname = Path(train_path).stem[: -len('_train')]
    print(dataname)
    dataname_test = Path(test_path).stem[: -len('_test')]
    assert dataname == dataname_test, f'{dataname} != {dataname_test}'

    # Load data
    X_train, y_train, cols = open_csv(train_path)
    X_test, y_test, _ = open_csv(test_path)

    path_svm_json = os.path.join(path_output, 'alfa_svm', dataname + '_svm.json')
    if os.path.exists(path_svm_json):
        best_params = open_json(path_svm_json)
    else:
        # Tune parameters
        clf = SVC()
        random_search = RandomizedSearchCV(
            clf,
            param_distributions=SVM_PARAM_DICT,
            n_iter=N_ITER_SEARCH,
            cv=5,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        # Save SVM params as JSON
        to_json(best_params, path_svm_json)
    print('Best params:', best_params)

    # Train model
    clf = SVC(**best_params)
    clf.fit(X_train, y_train)

    # Generate poison labels
    acc_train_clean, acc_test_clean, acc_train_poison, acc_test_poison, path_poison_data_list = compute_and_save_flipped_data(
        X_train, y_train,
        X_test, y_test,
        clf,
        os.path.join(path_output, 'alfa_svm', dataname),
        cols,
        advx_range,
    )

    # Save results
    data = {
        'Data': np.tile(dataname, reps=len(advx_range)),
        'Path.Train': np.tile(train_path, reps=len(advx_range)),
        'Path.Poison': path_poison_data_list,
        'Path.Test': np.tile(test_path, reps=len(advx_range)),
        'Rate': advx_range,
        'Train.Clean': acc_train_clean,
        'Test.Clean': acc_test_clean,
        'Train.Poison': acc_train_poison,
        'Test.Poison': acc_test_poison,
    }
    df = pd.DataFrame(data)
    if os.path.exists(os.path.join(path_output, 'synth_alfa_svm_score.csv')):
        df.to_csv(os.path.join(path_output, 'synth_alfa_svm_score.csv'), mode='a', header=False, index=False)
    else:
        df.to_csv(os.path.join(path_output, 'synth_alfa_svm_score.csv'), index=False)
