import os, sys
from os.path import abspath

from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris

import numpy as np

from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning.poisoning_attack_svm import PoisoningAttackSVM