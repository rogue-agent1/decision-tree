#!/usr/bin/env python3
"""decision_tree - ID3/CART decision tree with Gini and entropy splitting."""
import sys, json, math
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature; self.threshold = threshold
        self.left = left; self.right = right; self.label = label

def gini(labels):
    n = len(labels); counts = Counter(labels)
    return 1 - sum((c/n)**2 for c in counts.values())

def entropy(labels):
    n = len(labels); counts = Counter(labels)
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c > 0)

def best_split(X, y, criterion="gini"):
    score_fn = gini if criterion == "gini" else entropy
    best = {"score": float('inf'), "feat": None, "thresh": None}
    n = len(y)
    for f in range(len(X[0])):
        vals = sorted(set(row[f] for row in X))
        for i in range(len(vals)-1):
            thresh = (vals[i]+vals[i+1])/2
            left_y = [y[j] for j in range(n) if X[j][f] <= thresh]
            right_y = [y[j] for j in range(n) if X[j][f] > thresh]
            if not left_y or not right_y: continue
            score = len(left_y)/n*score_fn(left_y) + len(right_y)/n*score_fn(right_y)
            if score < best["score"]:
                best = {"score": score, "feat": f, "thresh": thresh}
    return best

def build_tree(X, y, depth=0, max_depth=5, min_samples=2, criterion="gini"):
    if len(set(y)) == 1: return Node(label=y[0])
    if depth >= max_depth or len(y) < min_samples:
        return Node(label=Counter(y).most_common(1)[0][0])
    split = best_split(X, y, criterion)
    if split["feat"] is None: return Node(label=Counter(y).most_common(1)[0][0])
    left_idx = [i for i in range(len(y)) if X[i][split["feat"]] <= split["thresh"]]
    right_idx = [i for i in range(len(y)) if X[i][split["feat"]] > split["thresh"]]
    left = build_tree([X[i] for i in left_idx], [y[i] for i in left_idx], depth+1, max_depth, min_samples, criterion)
    right = build_tree([X[i] for i in right_idx], [y[i] for i in right_idx], depth+1, max_depth, min_samples, criterion)
    return Node(feature=split["feat"], threshold=split["thresh"], left=left, right=right)

def predict(node, x):
    if node.label is not None: return node.label
    if x[node.feature] <= node.threshold: return predict(node.left, x)
    return predict(node.right, x)

def accuracy(tree, X, y):
    return sum(1 for i in range(len(y)) if predict(tree, X[i]) == y[i]) / len(y)

def main():
    X = [[2,3],[1,1],[3,2],[6,5],[7,8],[6,7],[8,6],[7,5],[3,6],[4,7]]
    y = [0,0,0,1,1,1,1,1,0,0]
    print("Decision tree demo\n")
    for crit in ["gini", "entropy"]:
        tree = build_tree(X, y, max_depth=3, criterion=crit)
        acc = accuracy(tree, X, y)
        preds = [predict(tree, x) for x in [[4,4],[8,7],[1,2]]]
        print(f"  {crit:8s}: accuracy={acc*100:.0f}%, predictions={preds}")

if __name__ == "__main__":
    main()
