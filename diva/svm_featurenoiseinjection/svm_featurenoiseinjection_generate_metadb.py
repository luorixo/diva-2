import argparse
import glob
import os
import time
import warnings
from pathlib import Path

from pymfe.mfe import MFE
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform

from utils.utils import create_dir, open_csv, to_csv

# Ignore warnings from optimization.
warnings.filterwarnings('ignore')

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

# POISON DATASETS WITH FEATURE NOISE INJECTION
def inject_feature_noise(X_train, rate, noise_level=3.0):
    X_noisy = X_train.copy()
    n_noisy = int(len(X_train) * rate)
    noisy_indices = np.random.choice(len(X_train), size=n_noisy, replace=False)
    # Generate noise for the selected samples
    noise = np.random.normal(0, noise_level, size=X_train.shape)
    # Add noise only to the selected samples
    X_noisy[noisy_indices] += noise[noisy_indices]
    return X_noisy

def compute_and_save_noisy_data(X_train, y_train, X_test, y_test, clf, path_output_base, cols, noise_rate_range):
    acc_train_clean = clf.score(X_train, y_train)
    acc_test_clean = clf.score(X_test, y_test)

    accuracy_train_clean = [acc_train_clean] * len(noise_rate_range)
    accuracy_test_clean = [acc_test_clean] * len(noise_rate_range)
    accuracy_train_noisy = []
    accuracy_test_noisy = []
    path_noisy_data_list = []

    for rate in noise_rate_range:
        path_noisy_data = '{}_featurenoiseinjection_svm_{:.2f}.csv'.format(path_output_base, np.round(rate, 2))
        try:
            if os.path.exists(path_noisy_data):
                X_train_noisy, y_train_noisy, _ = open_csv(path_noisy_data)
            else:
                X_train_noisy = inject_feature_noise(X_train, rate)
                y_train_noisy = y_train.copy()
                to_csv(X_train_noisy, y_train_noisy, cols, path_noisy_data)
            svm_params = clf.get_params()
            clf_noisy = SVC(**svm_params)
            clf_noisy.fit(X_train_noisy, y_train_noisy)
            acc_train_noisy = clf_noisy.score(X_train_noisy, y_train_noisy)
            acc_test_noisy = clf_noisy.score(X_test, y_test)
        except Exception as e:
            print(e)
            acc_train_noisy = 0
            acc_test_noisy = 0
        print('Noise Rate [{:.2f}%] - Acc  Noisy Train: {:.2f}%  Test Set: {:.2f}%'.format(rate * 100, acc_train_noisy * 100, acc_test_noisy * 100))
        path_noisy_data_list.append(path_noisy_data)
        accuracy_train_noisy.append(acc_train_noisy)
        accuracy_test_noisy.append(acc_test_noisy)

    return (accuracy_train_clean, accuracy_test_clean, accuracy_train_noisy, accuracy_test_noisy, path_noisy_data_list)

def feature_noise_poison(file_paths, noise_rate_range, path_output):
    for file_path in file_paths:
        # Load data
        X_train, y_train, cols = open_csv(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

        dataname = Path(file_path).stem
        create_dir(os.path.join(path_output, 'feature_noise_svm'))

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

        # Generate noisy data and evaluate
        acc_train_clean, acc_test_clean, acc_train_noisy, acc_test_noisy, path_noisy_data_list = compute_and_save_noisy_data(
            X_train, y_train,
            X_test, y_test,
            clf,
            os.path.join(path_output, 'feature_noise_svm', dataname),
            cols,
            noise_rate_range,
        )

        # Save results
        data = {
            'Data': np.tile(dataname, reps=len(noise_rate_range)),
            'Path.Poison': path_noisy_data_list,
            'Rate': noise_rate_range,
            'Train.Clean': acc_train_clean,
            'Test.Clean': acc_test_clean,
            'Train.Poison': acc_train_noisy,
            'Test.Poison': acc_test_noisy,
        }
        df = pd.DataFrame(data)
        csv_output_path = os.path.join(path_output, 'synth_feature_noise_svm_score.csv')
        if os.path.exists(csv_output_path):
            df.to_csv(csv_output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_output_path, index=False)

# EXTRACT COMPLEXITY MEASURES FROM CLEAN/POISONED DATASETS
def extract_complexity_measures(input_path):
    # Find all poisoned datasets
    poisoned_files = glob.glob(os.path.join(input_path, '*.csv'))

    results = []
    for file in poisoned_files:
        print(f'Computing c-measures for: {file}...')
        # Load the poisoned dataset
        data = pd.read_csv(file)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Initialize MFE (meta feature extractor) with complexity measures
        mfe = MFE(groups=["complexity"])

        # Fit the MFE model to the data
        mfe.fit(X, y)

        # Extract meta-features
        features, values = mfe.extract()

        # Collect results
        result = {'file': os.path.basename(file)}
        result.update(dict(zip(features, values)))
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

def extract_key(filename):
    # Extract the part that matches the pattern in the file names
    filename = os.path.basename(filename)  # Get the file name without the path
    return '_'.join(filename.split('_')[:10])  # Return the core file name part

def make_metadb(csv_path, cmeasure_dataframe, output_path):
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"No CSV found at {csv_path}. Creating a new CSV at {output_path}.")
        cmeasure_dataframe.to_csv(csv_path, index=False)
        return

    # Read the CSV file
    csv_data = pd.read_csv(csv_path)

    # Extract the key from the 'Path.Poison' column for matching
    csv_data['key'] = csv_data['Path.Poison'].apply(extract_key)

    # Adjust the 'file' column in cmeasure_dataframe to match the format of 'key'
    cmeasure_dataframe['key'] = cmeasure_dataframe['file'].apply(extract_key)

    # Log data to debug
    print("CSV Data 'key' column:", csv_data['key'].head())
    print("DataFrame 'key' column:", cmeasure_dataframe['key'].head())

    # Merge the two datasets on the extracted key
    merged_data = pd.merge(csv_data, cmeasure_dataframe, on='key', how='inner')

    if merged_data.empty:
        print("No matching data found for merging. Check the 'key' and 'file' columns.")

    # Save the merged data to a new CSV file
    merged_data.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=1, type=int, help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str, help='The output folder.')
    parser.add_argument('-s', '--step', type=float, default=0.05, help='Spacing between values for noise rates. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.41, help='End of interval for noise rates. Default=0.41')
    args = parser.parse_args()

    # Step 1: Generate synthetic datasets and save them to CSV files - default: data/synth
    generated_files = generate_synthetic_data(args.nSets, args.folder)

    # Step 2: Apply Feature Noise Injection on the generated datasets
    noise_rate_range = np.arange(0, args.max, args.step)
    feature_noise_poison(generated_files, noise_rate_range, 'poisoned_data')

    # Step 3: Compute complexity measures from clean/poisoned files
    complexity_measures_df = extract_complexity_measures('poisoned_data/feature_noise_svm')
    print(complexity_measures_df)

    # Step 4: Make meta database from information gathered
    csv_path = 'poisoned_data/synth_feature_noise_svm_score.csv'
    make_metadb(csv_path, complexity_measures_df, 'meta_database_feature_noise_svm.csv')
