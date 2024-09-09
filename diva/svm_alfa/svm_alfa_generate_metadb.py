import os
import glob
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_sets, folder):
    N_SAMPLES = np.arange(1000, 3001, 200)
    N_CLASSES = 2  # Number of classes

    # Create directory
    data_path = os.path.join('data', folder)
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
    
    # Replace if we need more sets than available parameters
    replace = len(param_sets) < n_sets
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=replace)

    for i in selected_indices:
        # Set a new random state for each parameter set
        param_sets[i]['random_state'] = np.random.randint(1000, np.iinfo(np.int16).max)
        
        # Generate data
        X, y = make_classification(**param_sets[i])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to DataFrame and save
        feature_names = ['x' + str(j) for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        df['y'] = y.astype(np.int32)
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=5, type=int, help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str, help='The output folder.')
    args = parser.parse_args()
    
    # Generate synthetic datasets
    generate_synthetic_data(args.nSets, args.folder)
