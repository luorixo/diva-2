import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class DifficultyGenerator:
    def __init__(self, n_sets, folder):
        self.n_sets = n_sets
        self.folder = folder
        self.N_SAMPLES = np.arange(1000, 3001, 200)
        self.N_CLASSES = 2  # Number of classes
        self.N_DIFFICULTY = 3
        self.DIFFICULTY_RANGE = [0.7, 0.9]

    def save_data(self, df, file_name, data_path, difficulty, postfix):
        path_output = os.path.join(data_path, f'{difficulty}_{file_name}_{postfix}.csv')
        df.to_csv(path_output, index=False)

    def gen_synth_data(self, data_path, param, bins):
        X, y = make_classification(**param)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        feature_names = ['x' + str(i) for i in range(1, X.shape[1] + 1)]

        # To dataframe
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        df['y'] = y
        df['y'] = df['y'].astype('category')

        # Format name based on params
        file_name = 'f{:02d}_i{:02d}_r{:02d}_c{:02d}_w{:.0f}_n{}'.format(
            param['n_features'],
            param['n_informative'],
            param['n_redundant'],
            param['n_clusters_per_class'],
            param['weights'][0] * 10,
            param['n_samples'])
        data_list = glob.glob(os.path.join(data_path, file_name + '*.csv'))
        postfix = str(len(data_list) + 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = SVC()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        if acc <= self.DIFFICULTY_RANGE[0] and bins[2] > 0:  # Hard
            self.save_data(df, file_name, data_path, 'Hard', postfix)
            bins[2] -= 1
            print(f'Hard:   {bins[2]}')
        elif acc <= self.DIFFICULTY_RANGE[1] and bins[1] > 0:  # Normal
            self.save_data(df, file_name, data_path, 'Normal', postfix)
            bins[1] -= 1
            print(f'Normal: {bins[1]}')
        elif acc > self.DIFFICULTY_RANGE[1] and bins[0] > 0:  # Easy
            self.save_data(df, file_name, data_path, 'Easy', postfix)
            bins[0] -= 1
            print(f'Easy:   {bins[0]}')
        else:
            print(f'Ditch {file_name}')

    def synth_data_grid(self):
        n_per_bin = self.n_sets // self.N_DIFFICULTY
        bins = n_per_bin * np.ones(self.N_DIFFICULTY)

        # Create directory
        data_path = os.path.join('data', self.folder)
        if not os.path.exists(data_path):
            print('Create path:', data_path)
            path = Path(data_path)
            path.mkdir(parents=True)

        grid = []
        for f in range(4, 31):
            grid.append({
                'n_samples': self.N_SAMPLES,
                'n_classes': [self.N_CLASSES],
                'n_features': [f],
                'n_repeated': [0],
                'n_informative': np.arange(math.ceil(f / 2), f + 1),
                'weights': [[0.4], [0.5], [0.6]]})
        param_sets = list(ParameterGrid(grid))
        print('# of parameter sets:', len(param_sets))
        for i in range(len(param_sets)):
            param_sets[i]['n_redundant'] = np.random.randint(
                0, high=param_sets[i]['n_features'] + 1 - param_sets[i]['n_informative'])
            param_sets[i]['n_clusters_per_class'] = np.random.randint(
                1, param_sets[i]['n_informative'])

        # Replace iff we need more sets than it has.
        replace = len(param_sets) < self.n_sets
        selected_indices = np.random.choice(
            len(param_sets), self.n_sets, replace=replace)
        for i in selected_indices:
            # This ensure the generator gets a new RND seed everytime
            param_sets[i]['random_state'] = np.random.randint(
                1000, np.iinfo(np.int16).max)
            self.gen_synth_data(data_path, param_sets[i], bins)
            if np.sum(bins) <= 0:
                print('Generation completed!')
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=1000, type=int,
                        help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str,
                        help='The output folder.')
    args = parser.parse_args()
    
    # Create an instance of DifficultyGenerator
    generator = DifficultyGenerator(args.nSets, args.folder)
    
    # Generate synthetic datasets
    generator.synth_data_grid()
