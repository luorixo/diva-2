import argparse
from data_generators.difficulty_generator import DifficultyGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=100, type=int,
                        help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str,
                        help='The output folder.')
    args = parser.parse_args()
    
    # Create an instance of DifficultyGenerator
    generator = DifficultyGenerator(args.nSets, args.folder)
    
    # Generate synthetic datasets
    generator.synth_data_grid()

if __name__ == '__main__':
    main()
