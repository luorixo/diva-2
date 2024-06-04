import argparse
import glob
import os
from pathlib import Path

from datagenerators.difficultygenerator import DifficultyGenerator
from alfapoison import alfa_poison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=100, type=int,
                        help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str,
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
    max = args.max

    advxrange = np.arange(0, max, step)

    print('Path:', filepath)
    print('Range:', advxrange)

    trainlist = sorted(glob.glob(os.path.join(filepath, 'train', '.csv')))
    test_list = sorted(glob.glob(os.path.join(filepath, 'test', '.csv')))
    assert len(trainlist) == len(testlist)
    print('Found {} datasets'.format(len(train_list)))

    for train, test in zip(train_list, test_list):
        dataset = {'train': train, 'test': test}
        alfa_poison(dataset, advx_range, output)

if __name == '__main':
    main()