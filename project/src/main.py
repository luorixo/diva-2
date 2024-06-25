import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

from data_generators.difficulty_generator import DifficultyGenerator
from poisoners.alfa_poisoner import alfa_poison
from meta_database.make_metadb import make_metadb
from utils.test_train_split import test_train_split
from meta_database.extract_complexity_measures import extract_complexity_measures
from memento import Config, Context, Memento

def poison_experiment(context: Context, config: Config):

    # Grab values from matrix
    data = config.data
    output = config.output
    poison_method = config.poison_method
    step = config.poison_step
    max_ = config.poison_max

    dataset = {"train": data[0], "test": data[1]}
    advx_range = np.arange(0, max_, step)

    return poison_method(dataset, advx_range, output)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=8, type=int,
                        help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='', type=str,
                        help='The output folder.')
    parser.add_argument('-o', '--output', type=str, default='results/synth',
                        help='The output path for scores.')
    parser.add_argument('-s', '--step', type=float, default=0.05,
                        help='Spacing between values for poisoning rates. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.41,
                        help='End of interval for poisoning rates. Default=0.41')
    args = parser.parse_args()

    # Generate synthetic datasets
    generator = DifficultyGenerator(args.nSets, args.folder)
    generator.synth_data_grid()

    # Perform ALFA poisoning attack
    filepath = str(Path(args.folder).absolute())
    output = str(Path(args.output).absolute())

    data_path = os.path.join(filepath, 'data')
    split_path = os.path.join(filepath, 'split')
    test_train_split(data_path, split_path, 0.2)

    train_list = sorted(glob.glob(os.path.join(split_path, 'train', '*.csv')))
    test_list = sorted(glob.glob(os.path.join(split_path, 'test', '*.csv')))
    assert len(train_list) == len(test_list)

    print('Found {} datasets'.format(len(train_list)))

    data = list(zip(train_list, test_list))

    matrix_poison = {
        "parameters": {
            "data": data,
            "output": [output],
            "poison_method": [alfa_poison],
            "poison_max": [args.max],
            "poison_step": [args.step]
        },
    }

    Memento(poison_experiment).run(matrix_poison)

    # Extract complexity measures from poisoned datasets
    # directory hard coded for now
    complexity_measures_df = extract_complexity_measures(os.path.join(output, 'alfa_svm'))

    # Print the complexity measures
    print(complexity_measures_df)

    # Create metadata database
    csv_path = os.path.join(output, 'synth_alfa_svm_score.csv')
    make_metadb(csv_path, complexity_measures_df, os.path.join(output, 'metadata.csv'))
if __name__ == '__main__':
    main()
