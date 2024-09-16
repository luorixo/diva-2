from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
from pathlib import Path
import os, sys
from venv import logger
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import OneHotEncoder


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris, make_classification

import numpy as np
from matplotlib import pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning.poisoning_attack_svm import PoisoningAttackSVM
from sklearn.svm import SVC

np.random.seed(301)

def generate_synthetic_data(n_sets, folder):
    N_SAMPLES = np.arange(100, 200, 200)
    N_CLASSES = 2 

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
    
    return generated_files

def find_duplicates(x_train):
    """
    Returns an array of booleans that is true if that element was previously in the array

    :param x_train: training data
    :type x_train: `np.ndarray`
    :return: duplicates array
    :rtype: `np.ndarray`
    """
    dup = np.zeros(x_train.shape[0])
    for idx, x in enumerate(x_train):
        dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
    return dup

def get_adversarial_examples(x_train, y_train, x_val, y_val, kernel, poison_rate):
    # Create ART classifier for scikit-learn SVC
    art_classifier = SklearnClassifier(model=SVC(kernel=kernel), clip_values=(0, 10))
    art_classifier.fit(x_train, y_train)

    # Calculate the number of poisoned samples to generate
    num_samples_to_poison = int(len(x_train) * poison_rate)

    # Randomly select indices to poison
    attack_idxs = np.random.choice(len(x_train), num_samples_to_poison, replace=False)

    # Initialize attack samples and their corresponding labels
    init_attacks = np.copy(x_train[attack_idxs])
    y_attacks = np.array([1, 1]) - np.copy(y_train[attack_idxs])

    # Perform the poisoning attack on the selected samples
    attack = PoisoningAttackSVM(art_classifier, 0.001, 1.0, x_train, y_train, x_val, y_val, max_iter=100)
    poisoned_samples, poisoned_labels = attack.poison(init_attacks, y=y_attacks)

    return poisoned_samples, poisoned_labels, art_classifier

def extract_and_append_complexity_measures(poison_data, poison_labels, dataset_name, poisoning_type, rate,
                                           train_clean_acc, test_clean_acc,
                                           train_poison_acc, test_poison_acc,
                                           output_csv_path):
    # Initialize MFE (meta feature extractor) with complexity measures
    mfe = MFE(groups=["complexity"])

    # Fit the MFE model to the data
    mfe.fit(poison_data, poison_labels)

    # Extract meta-features
    features, values = mfe.extract()

    # Prepare the result as a dictionary
    result = {
        'Data': f'{poisoning_type}_size{len(poison_data)}_{dataset_name}',
        'Rate': rate,
        'Train.Clean': train_clean_acc,
        'Test.Clean': test_clean_acc,
        'Train.Poison': train_poison_acc,
        'Test.Poison': test_poison_acc
    }
    # Add complexity measures to the result
    result.update(dict(zip(features, values)))

    # Convert the result to a DataFrame
    result_df = pd.DataFrame([result])

    # Append the result to the existing CSV
    if not os.path.isfile(output_csv_path):
        result_df.to_csv(output_csv_path, index=False)
    else:
        result_df.to_csv(output_csv_path, mode='a', header=False, index=False)

    print(f'Appended complexity measures for {dataset_name} to {output_csv_path}')

def open_csv(path_data, label_name='y'):
    """Read data from a CSV file, return X, y and column names."""
    logger.info('Load from:', path_data)
    df_data = pd.read_csv(path_data)
    y = df_data[label_name].to_numpy()
    df_data = df_data.drop([label_name], axis=1)
    cols = df_data.columns
    X = df_data.to_numpy()
    return X, y, cols
def poison_and_extract(train_data, test_data, train_labels, test_labels):
    # Define the kernel type
    kernel = 'linear'  # One of ['linear', 'poly', 'rbf']

    # Check if the labels are one-hot encoded
    if len(train_labels.shape) == 1:
        # If labels are in integer format, one-hot encode them
        encoder = OneHotEncoder(sparse_output=True)  # Using dense output for compatibility
        train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))  # Fit on training data
        test_labels = encoder.transform(test_labels.reshape(-1, 1))  # Transform test data using the same encoder

    train_labels = train_labels.toarray()  # Convert sparse matrix to dense
    test_labels = test_labels.toarray()    # Convert sparse matrix to dense

    # Train the clean model without poisoning
    clean_model = SVC(kernel=kernel)
    art_clean = SklearnClassifier(clean_model)

    # Fit the model with the properly encoded labels
    art_clean.fit(x=train_data, y=train_labels)

    # Calculate accuracies for the clean model (apply argmax to convert to class labels)
    train_pred = np.argmax(art_clean.predict(train_data), axis=1)
    test_pred = np.argmax(art_clean.predict(test_data), axis=1)
    
    clean_acc_train = np.mean(train_pred == np.argmax(train_labels, axis=1))  # Use argmax to get actual labels
    clean_acc_test = np.mean(test_pred == np.argmax(test_labels, axis=1))

    # Flatten the clean labels for complexity measure extraction (but not for model training)
    train_labels_flat = train_labels.argmax(axis=1)

    # Loop through different poisoning rates
    for poison_rate in range(0, 45, 5):  # 0% to 40% in increments of 5%
        print(f"Processing poison_rate: {poison_rate}%")

        if poison_rate == 0:
            # No poisoning, just use the clean data
            poisoned_train_data = train_data
            poisoned_train_labels = train_labels
            poisoned_train_labels_flat = train_labels_flat
            poison_acc_train = clean_acc_train
            poison_acc_test = clean_acc_test
        else:
            # Get poisoned examples and the model after poisoning
            poisoned_samples, poisoned_labels, poisoned_model = get_adversarial_examples(
                train_data, train_labels, test_data, test_labels, kernel, poison_rate / 100.0
            )

            # Combine original training data with poisoned samples
            poisoned_train_data = np.vstack([train_data, poisoned_samples])
            poisoned_train_labels = np.vstack([train_labels, poisoned_labels])

            # Flatten the poisoned labels for complexity measure extraction
            poisoned_train_labels_flat = poisoned_train_labels.argmax(axis=1)

            # Re-initialize the model before re-training
            poisoned_model = SVC(kernel=kernel)
            art_poisoned = SklearnClassifier(poisoned_model)

            # Re-train the model with poisoned data (not flattening the labels)
            art_poisoned.fit(x=poisoned_train_data, y=poisoned_train_labels)

            # Calculate accuracies for the poisoned model (use argmax to compare integer labels)
            poison_train_pred = np.argmax(art_poisoned.predict(train_data), axis=1)
            poison_test_pred = np.argmax(art_poisoned.predict(test_data), axis=1)

            poison_acc_train = np.mean(poison_train_pred == np.argmax(train_labels, axis=1))
            poison_acc_test = np.mean(poison_test_pred == np.argmax(test_labels, axis=1))

        # Assuming train_data and train_labels are your datasets
        dataset_name = 'synth'  # Replace with the actual dataset name
        poisoning_type = 'PoisSVM'  # The type of poisoning attack
        output_csv_path = 'metadata.csv'  # The path to your existing CSV file

        # Call the function to extract complexity measures and append to the CSV
        extract_and_append_complexity_measures(poisoned_train_data, poisoned_train_labels_flat, dataset_name, poisoning_type, poison_rate / 100.0,
                                            clean_acc_train, clean_acc_test,
                                            poison_acc_train, poison_acc_test,
                                            output_csv_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=100, type=int,
                        help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='', type=str,
                        help='The output folder.')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='The output path for scores.')
    parser.add_argument('-s', '--step', type=float, default=0.05,
                        help='Spacing between values for poisoning rates. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.41,
                        help='End of interval for poisoning rates. Default=0.41')
    args = parser.parse_args()

    generated_files = generate_synthetic_data(args.nSets, args.folder)

    for file in generated_files:
        x, y, col = open_csv(file)
        X_train, X_test, y_train, y_test = train_test_split(x, y)
        poison_and_extract(X_train, X_test, y_train, y_test)
        
if __name__ == '__main__':
    main()