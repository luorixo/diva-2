import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from pymfe.mfe import MFE

# Constants
RANDOM_SEED = 100
PCA_COMPONENTS = 10  # Adjusted to your dataset
N_NEIGHBORS = 5
MIN_SAMPLES = 1000
MAX_SAMPLES = 2000
EPSILON = 0.1  # Perturbation magnitude
STEP_SIZE = 0.01  # Step size for perturbations

def generate_synthetic_data(n_sets, folder):
    N_SAMPLES = np.arange(MIN_SAMPLES, MAX_SAMPLES + 1, 20)
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
        param_sets[i]['n_redundant'] = np.random.randint(
            0, high=param_sets[i]['n_features'] + 1 - param_sets[i]['n_informative'])
        param_sets[i]['n_clusters_per_class'] = np.random.randint(
            1, param_sets[i]['n_informative'])

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


def load_and_preprocess_data(file):
    # Load your dataset here
    df = pd.read_csv(file)  # Replace with your dataset path

    # Ensure the last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    normalizer = Normalizer()
    X_normalized = normalizer.fit_transform(X_scaled)
    pca = PCA(n_components=min(PCA_COMPONENTS, X_normalized.shape[1]), random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_normalized)

    # Construct feature similarity graph
    graph = kneighbors_graph(X_pca, N_NEIGHBORS, mode='connectivity', include_self=True)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return X_pca, y, labels, n_components


def clean_label_attack(X_train, y_train, attack_class, attack_rate):
    """
    Implements a clean-label backdoor attack by perturbing samples from the attack_class.

    Args:
        X_train (ndarray): Training data.
        y_train (ndarray): Training labels.
        attack_class (int): The class to target for the backdoor.
        attack_rate (float): Proportion of attack_class samples to poison (e.g., 0.004 for 0.4%).

    Returns:
        X_poisoned (ndarray): Poisoned training data.
        y_poisoned (ndarray): Poisoned training labels (unchanged).
    """
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    # Identify indices of the attack_class
    attack_class_indices = np.where(y_train == attack_class)[0]

    if len(attack_class_indices) == 0:
        raise ValueError(f"No samples found for attack_class {attack_class}")

    num_attack_points = int(attack_rate * len(attack_class_indices))
    if num_attack_points == 0 and attack_rate > 0:
        num_attack_points = 1

    # Ensure num_attack_points does not exceed the number of attack_class samples
    num_attack_points = min(num_attack_points, len(attack_class_indices))

    # Randomly select samples to poison
    poisoned_indices = np.random.choice(attack_class_indices, num_attack_points, replace=False)

    # Define a backdoor pattern (a small perturbation vector)
    backdoor_pattern = np.full(X_train.shape[1], EPSILON)  # Small perturbation

    # Apply perturbations
    X_poisoned[poisoned_indices] += backdoor_pattern

    return X_poisoned, y_poisoned


# Function to train SVM
def train_svm(X_train, y_train, C=10, gamma=0.01):
    """
    Trains an SVM with specified hyperparameters.
    """
    svm = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    return svm

# Function to evaluate and collect results
def evaluate(svm, X_test, y_test):
    """
    Evaluates the SVM and returns the metrics.
    """
    y_pred = svm.predict(X_test)

    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return accuracy, balanced_acc, precision, recall, f1

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
    key = '_'.join(filename.split('_')[:-2])  # Remove the last parts (method and rate)
    return key

def make_metadb(csv_path, cmeasure_dataframe, output_path):
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"No CSV found at {csv_path}. Creating a new CSV at {output_path}.")
        cmeasure_dataframe.to_csv(output_path, index=False)
        return

    # Read the CSV file
    csv_data = pd.read_csv(csv_path)

    # Extract the key from the 'Path.Poison' column for matching
    csv_data['key'] = csv_data['Path.Poison'].apply(extract_key)

    # Adjust the 'file' column in cmeasure_dataframe to match the format of 'key'
    cmeasure_dataframe['key'] = cmeasure_dataframe['file'].apply(extract_key)

    # Merge the two datasets on the extracted key and Rate
    merged_data = pd.merge(csv_data, cmeasure_dataframe, on=['key'], how='inner')

    # Remove duplicate entries based on 'Path.Poison'
    merged_data = merged_data.drop_duplicates(subset=['Path.Poison'])

    if merged_data.empty:
        print("No matching data found for merging. Check the 'key' and 'file' columns.")

    # Save the merged data to a new CSV file
    merged_data.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

def clean_label_poison(files, attack_percentages, base_output_folder):
    # Define the base output folder for saving poisoned data and CSV results
    os.makedirs(base_output_folder, exist_ok=True)

    # CSV to store SVM scores
    csv_output_path = os.path.join(base_output_folder, 'synth_cleanlabel_svm_score.csv')

    clean_label_attack_folder = os.path.join(base_output_folder, 'clean_label')
    os.makedirs(clean_label_attack_folder, exist_ok=True)

    for file in files:
        X, y, _, _ = load_and_preprocess_data(file)

        # Split into training and testing sets for evaluation purposes
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        # Extract base file name without postfix
        base_file_name = '_'.join(os.path.basename(file).split('_')[:-1])  # Remove postfix

        # Initialize a list to store results
        results = []

        # Initialize SVM trained on clean data once (0% attack)
        print("\nStandard SVM Results (Clean Data):")
        svm_clean = train_svm(X_train_full, y_train_full)
        acc_train_clean, _, _, _, _ = evaluate(svm_clean, X_train_full, y_train_full)
        acc_test_clean, _, _, _, _ = evaluate(svm_clean, X_test, y_test)

        # Save clean training data with specified naming convention
        rate_str = "{:.2f}".format(0.00)
        poison_file_name = f"{base_file_name}_clean_svm_{rate_str}.csv"
        poison_data_path = os.path.join(clean_label_attack_folder, poison_file_name)
        pd.DataFrame(np.hstack([X_train_full, y_train_full.reshape(-1, 1)]),
                     columns=[f"feature_{i}" for i in range(X_train_full.shape[1])] + ["label"]).to_csv(poison_data_path, index=False)
        print(f"Saved clean data to: {poison_data_path}")

        # Append results with Path.Poison
        results.append({
            'Data': os.path.basename(file),
            'Path.Poison': poison_data_path,
            'Rate': 0.00,
            'Train.Clean': acc_train_clean,
            'Test.Clean': acc_test_clean,
            'Train.Poison': acc_train_clean,  # Same as clean
            'Test.Poison': acc_test_clean    # Same as clean
        })

        for attack_percentage in attack_percentages[1:]:  # Skip 0% as it's already processed
            rate = attack_percentage
            print(f"\nProcessing attack percentage: {rate*100:.2f}%")
            num_attack_points = int(rate * len(X_train_full))

            # Ensure at least one attack point if attack_percentage > 0 and num_attack_points == 0
            if num_attack_points == 0 and rate > 0:
                num_attack_points = 1

            # Clean-Label Backdoor Attack
            print(f"\nSVM Results (Poisoned Data - Clean-Label Backdoor Attack at {rate*100:.2f}%):")
            attack_class = 1  # The class to attack (e.g., 1)

            # Perform the backdoor attack
            X_poisoned, y_poisoned = clean_label_attack(
                X_train_full, y_train_full, attack_class, rate
            )

            # Retrain SVM on poisoned data
            svm_poisoned = train_svm(X_poisoned, y_poisoned)

            # Evaluate SVM on poisoned data
            acc_train_poison, _, _, _, _ = evaluate(svm_poisoned, X_poisoned, y_poisoned)
            acc_test_poison, _, _, _, _ = evaluate(svm_poisoned, X_test, y_test)

            # Save poisoned data with attack percentage in filename
            rate_str = "{:.2f}".format(rate)
            poison_file_name = f"{base_file_name}_backdoor_svm_{rate_str}.csv"
            poison_data_path = os.path.join(clean_label_attack_folder, poison_file_name)
            pd.DataFrame(np.hstack([X_poisoned, y_poisoned.reshape(-1, 1)]),
                         columns=[f"feature_{i}" for i in range(X_poisoned.shape[1])] + ["label"]).to_csv(poison_data_path, index=False)
            print(f"Saved poisoned data to: {poison_data_path}")

            # Append results with Path.Poison
            results.append({
                'Data': os.path.basename(file),
                'Path.Poison': poison_data_path,
                'Rate': rate,
                'Train.Clean': acc_train_clean,
                'Test.Clean': acc_test_clean,
                'Train.Poison': acc_train_poison,
                'Test.Poison': acc_test_poison
            })

        # Save results to CSV after processing each file
        df_results = pd.DataFrame(results)
        if os.path.exists(csv_output_path):
            df_results.to_csv(csv_output_path, mode='a', header=False, index=False)
        else:
            df_results.to_csv(csv_output_path, index=False)
        print(f"\nSaved SVM scores to: {csv_output_path}")

def main():
    num_files = 10  # Adjust the number of synthetic datasets to generate
    generated_files = generate_synthetic_data(num_files, folder="")

    # Define the base output folder for saving poisoned data and CSV results
    base_output_folder = 'poisoned_data'
    os.makedirs(base_output_folder, exist_ok=True)

    # CSV to store SVM scores
    csv_output_path = os.path.join(base_output_folder, 'synth_poissvm_svm_score.csv')

    attack_percentages = [0.0, 0.004, 0.015, 0.06, 0.25, 0.5]
    clean_label_poison(generated_files, attack_percentages, base_output_folder)

    # Extract complexity measures
    print("\nExtracting complexity measures...")
    complexity_measures_df = extract_complexity_measures(os.path.join(base_output_folder, 'clean_label'))

    # Make meta database from information gathered
    print("\nCreating meta database...")
    metadb_output_path = os.path.join("", 'meta_database_cleanlabel_svm.csv')
    make_metadb(csv_output_path, complexity_measures_df, metadb_output_path)

if __name__ == "__main__":
    main()
