import numpy as np
import matplotlib.pyplot as plt

from Lab01 import left_indices
from public_tests import *
from utils import *

# Helper functions
def compute_entropy(y):
    edible = np.sum(y)
    total = len(y)

    if total == 0:
        return 0

    p1 = edible / total

    if p1 == 0 or p1 == 1:
        return 0

    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

def split_dataset(X, node_indices, feature):
    left_indices, right_indices = [], []

    for index in node_indices:
        if X[index][feature] == 1:
            left_indices.append(index)
        else:
            right_indices.append(index)

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left  = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    w_left = len(y_left) / len(y_node)
    w_right = len(y_right) / len(y_node)

    information_gain = compute_entropy(y_node) - (w_left * compute_entropy(y_left) + w_right * compute_entropy(y_right))

    return information_gain

def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = -1

    biggest_information_gain = None
    for feature in range(num_features):
        information_gain = compute_information_gain(X, y, node_indices, feature)

        if information_gain <= 0:
            continue
        elif biggest_information_gain == None or information_gain > biggest_information_gain:
            biggest_information_gain = information_gain
            best_feature = feature

    return best_feature

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth=0):
    tree = []
    def _build_tree_recursive(_node_indices, _branch_name, _current_depth):
        if _current_depth == max_depth or len(set(y[_node_indices])) == 1 or len(_node_indices) == 0:
            return

        best_feature = get_best_split(X, y, _node_indices)

        if best_feature == -1:
            return

        left_indices, right_indices = split_dataset(X, _node_indices, best_feature)

        tree.append((left_indices, right_indices, best_feature))

        _build_tree_recursive(left_indices, "Left", _current_depth + 1)
        _build_tree_recursive(right_indices, "Right", _current_depth + 1)

    _build_tree_recursive(node_indices, branch_name, current_depth)
    return tree

# [IMPORT DATA]
# [Color] 1 = Brown; 0 = Red
# [Tapering] 1 = Tapering Stalk; 0 = Enlarging
# [Solitary] 1 = Yes; 0 = No

X_train = np.array([
# Color; Shape; Solitary

    [1,1,1],
    [1,0,1],
    [1,0,0],
    [1,0,0],
    [1,1,1],
    [0,1,1],
    [0,0,0],
    [1,0,1],
    [0,1,0],
    [1,0,0]
])

# 1 = Edible; 0 = Poisonous
y_train = np.array([1,1,0,0,1,0,0,1,1,0])


root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

tree = build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
print(tree)
# generate_tree_viz(root_indices, y_train, tree)
