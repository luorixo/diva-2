import argparse
import glob
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform

from utils.alfa import alfa
from utils.utils import create_dir, open_csv, open_json, to_csv, to_json, transform_label

# Ignore warnings from optimization.
warnings.filterwarnings('ignore')

ALFA_MAX_ITER = 5  # Number of iterations for ALFA.
N_ITER_SEARCH = 50  # Number of iterations for SVM parameter tuning.
SVM_PARAM_DICT = {
    'C': loguniform(0.01, 10),
    'gamma': loguniform(0.01, 10),
    'kernel': ['rbf'],
}

# GENERATE SYNTHETIC DATASETS
def generate_synthetic_data(n_sets, folder):
    N_SAMPLES = np.arange(1000, 3001, 200)
    N_CLASSES = 2  # Number of classes

    # Create directory
    data_path = os.path.join('clean_data', folder)
    if not os.path.exists(data_path):
        print('Create path:', data_path)
        path = Path(data_path)
        path.mkdir(parents=True)

    grid = []
    for f in range(4, 31):
        grid.append({
            'n_samples': N_SAMPLES,
            'n_classes': [N_CLASSES],
            'n_features': [f],
            'n_repeated': [0],
            'n_informative': np.arange(f // 2, f + 1),
            'weights': [[0.4], [0.5], [0.6]]})
    
    param_sets = list(ParameterGrid(grid))
    print('# of parameter sets:', len(param_sets))
    
    # Adjust redundant features and clusters per class for each parameter set
    for i in range(len(param_sets)):
        param_sets[i]['n_redundant'] = np.random.randint(0, high=param_sets[i]['n_features'] + 1 - param_sets[i]['n_informative'])
        param_sets[i]['n_clusters_per_class'] = np.random.randint(1, param_sets[i]['n_informative'])
    
    replace = len(param_sets) < n_sets
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=replace)

    generated_files = []  # Keep track of generated files

    for i in selected_indices:
        param_sets[i]['random_state'] = np.random.randint(1000, np.iinfo(np.int16).max)
        
        X, y = make_classification(**param_sets[i])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        feature_names = ['x' + str(j) for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        df['y'] = y.astype(np.int32)  # Convert y to integers
        
        file_name = 'f{:02d}_i{:02d}_r{:02d}_c{:02d}_w{:.0f}_n{}'.format(
            param_sets[i]['n_features'],
            param_sets[i]['n_informative'],
            param_sets[i]['n_redundant'],
            param_sets[i]['n_clusters_per_class'],
            param_sets[i]['weights'][0] * 10,
            param_sets[i]['n_samples'])
        
        data_list = glob.glob(os.path.join(data_path, file_name + '*.csv'))
        postfix = str(len(data_list) + 1)
        
        output_path = os.path.join(data_path, f'{file_name}_{postfix}.csv')
        df.to_csv(output_path, index=False)
        print(f'Saved: {output_path}')

        generated_files.append(output_path)  # Store the path of each generated file
    
    return generated_files  # Return the list of generated files for further processing

def get_y_flip(X_train, y_train, rate, svc):
    if rate == 0:
        return y_train

    y_train = transform_label(y_train, target=-1)
    y_flip = alfa(X_train, y_train, rate, svc_params=svc.get_params(), max_iter=ALFA_MAX_ITER)
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
    
    return (accuracy_train_clean, accuracy_test_clean, accuracy_train_poison, accuracy_test_poison, path_poison_data_list)

def alfa_poison(file_paths, advx_range, path_output):
    for file_path in file_paths:
        # Load data
        X_train, y_train, cols = open_csv(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

        dataname = Path(file_path).stem
        create_dir(os.path.join(path_output, 'alfa_svm'))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=2, type=int, help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str, help='The output folder.')
    parser.add_argument('-s', '--step', type=float, default=0.05, help='Spacing between values for poisoning rates. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.41, help='End of interval for poisoning rates. Default=0.41')
    args = parser.parse_args()

    # Step 1: Generate synthetic datasets and save them to CSV files - default: data/synth
    generated_files = generate_synthetic_data(args.nSets, args.folder)

    # Step 3: Apply ALFA poisoning on the generated datasets
    advx_range = np.arange(0, args.max, args.step)
    alfa_poison(generated_files, advx_range, 'poisoned_data')
    