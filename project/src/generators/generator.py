from abc import ABC, abstractmethod

class Generator(ABC):
    def __init__(self, n_samples, n_classes, n_sets, n_difficulty):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_sets = n_sets
        self.n_difficulty = n_difficulty

    @abstractmethod
    def gen_synth_data(self):
        pass

    @abstractmethod
    def synth_data_grid(self, data, filepath):
        pass