
import argparse
import os
from pathlib import Path
from glob import glob
from datetime import datetime
from label_flip_revised.utils import open_csv
import pandas as pd
import logging

from label_flip_revised.flip_random import flip_random
logging.basicConfig(level=logging.INFO)

from memento import Config, Context, Memento
import numpy as np


def poison_dataset_experiment(context: Context, config: Config):
    path = config.data_path

    X, y, cols = open_csv(path) 
    y_poisoned = flip_random(y, 0.5)

    df = pd.DataFrame(X, columns=cols, dtype=np.float32)
    df['y'] = y_poisoned
    df['y'] = df['y'].astype('category')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path')
    args = parser.parse_args()
    filepath = str(Path(args.filepath).absolute())
    output = str(Path(args.output).absolute())

    path_list = sorted(glob(os.path.join(filepath, '*.csv')))

    datasets = {}

    matrix_poison = {
        "parameters": {
            "data_path": path_list,
            "output": [output],
            "poison_method": [flip_random],
            "poison_values": [0.5]
        },
    }

    results = Memento(poison_dataset_experiment).run(matrix_poison)
    for result in results:
        df = result.inner
        print(df) 
        output_file = os.path.join(output, f'poisoned_{os.path.basename(result.config.data_path)}')
        df.to_csv(output_file, index=False)
        print(f"Saved poisoned dataset to {output_file}")