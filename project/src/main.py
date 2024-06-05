import argparse
import glob
import os
from pathlib import Path
import numpy as np

from data_generators.difficulty_generator import DifficultyGenerator
from poisoners.alfa_poisoner import alfa_poison
from utils.test_train_split import test_train_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=100, type=int,
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
    step = args.step
    max_ = args.max

    advx_range = np.arange(0, max_, step)

    print('Path:', filepath)
    print('Range:', advx_range)

    data_path = os.path.join(filepath, 'data')
    split_path = os.path.join(filepath, 'split')
    test_train_split(data_path, split_path, 0.2)

    train_list = sorted(glob.glob(os.path.join(split_path, 'train', '*.csv')))
    test_list = sorted(glob.glob(os.path.join(split_path, 'test', '*.csv')))
    assert len(train_list) == len(test_list)
    print('Found {} datasets'.format(len(train_list)))

    for train, test in zip(train_list, test_list):
        dataset = {'train': train, 'test': test}
        alfa_poison(dataset, advx_range, output)

if __name__ == '__main__':
    main()
