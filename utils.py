import os
import random
import numpy as np
import torch
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

def set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_all_python_files(parent_dir="."):

    parent_dir = Path(parent_dir)
    py_files = list(parent_dir.rglob("*.py"))
    for py_file in py_files:
        if not "conda_env" in str(py_file):
            if not "mlruns" in str(py_file):
                mlflow.log_artifact(str(py_file))
                