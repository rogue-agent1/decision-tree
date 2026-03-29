#!/usr/bin/env python3
"""Decision tree classifier from scratch."""
import sys, csv, math, random
from collections import Counter

def entropy(labels):
    n = len(labels)
    if n == 0: return 0
    counts = Counter(labels)
    return -sum((c/n) * math.log2(c/n) for c in counts.values() if c > 0)

def gini(labels):
    n = len(labels)
    if n == 0: return 0
    counts = Counter(labels)
    return 1 - sum((c/n)**2 for c in counts.values())

class Node:
    def __init__(self):
        self.feature = None; self.threshold = None
        self.left = None; self.right = None; self.label = None; self.samples = 0

def best_split(X, y, criterion="gini"):
    best_gain = 0; best_feat = None; best_thresh = None
    score_fn = gini if criterion == "gini" else entropy
    parent_score = score_fn(y); n = len(y)
    for f in range(len(X[0])):
        vals = sorted(set(row[f] for row in X))
        for i in range(len(vals) - 1):
            thresh = (vals[i] + vals[i+1]) / 2
            left_y = [y[j] for j in range(n) if X[j][f] <= thresh]
            right_y = [y[j] for j in range(n) if X[j][f] > thresh]
            if not left_y or not right_y: continue
            gain = parent_score - (len(left_y)/n * score_fn(left_y) + len(right_y)/n * score_fn(right_y))
            if gain > best_gain: best_gain = gain; best_feat = f; best_thresh = thresh
    return best_feat, best_thresh, best_gain

def build_tree(X, y, max_depth=10, min_samples=2, depth=0):
    node = Node(); node.samples = len(y)
    node.label = Counter(y).most_common(1)[0][0]
    if depth >= max_depth or len(set(y)) == 1 or len(y) < min_samples: return node
    feat, thresh, gain = best_split(X, y)
    if feat is None or gain < 1e-7: return node
    node.feature = feat; node.threshold = thresh
    left_idx = [i for i in range(len(y)) if X[i][feat] <= thresh]
    right_idx = [i for i in range(len(y)) if X[i][feat] > thresh]
    node.left = build_tree([X[i] for i in left_idx], [y[i] for i in left_idx], max_depth, min_samples, depth+1)
    node.right = build_tree([X[i] for i in right_idx], [y[i] for i in right_idx], max_depth, min_samples, depth+1)
    return node

def predict(node, x):
    if node.feature is None: return node.label
    return predict(node.left if x[node.feature] <= node.threshold else node.right, x)

def print_tree(node, features=None, depth=0):
    sp = "  " * depth
    if node.feature is None:
        print(f"{sp}-> {node.label} (n={node.samples})"); return
    fname = features[node.feature] if features else f"f{node.feature}"
    print(f"{sp}{fname} <= {node.threshold:.3f}?")
    print_tree(node.left, features, depth + 1)
    print(f"{sp}{fname} > {node.threshold:.3f}?")
    print_tree(node.right, features, depth + 1)

def demo():
    print("=== Decision Tree Demo (Iris-like) ===")
    random.seed(42)
    X = [[random.gauss(m, 0.5) for m in center] for center in [[1,1],[3,3],[1,3]] for _ in range(30)]
    y = [c for c in ["A","B","C"] for _ in range(30)]
    idx = list(range(len(y))); random.shuffle(idx)
    X = [X[i] for i in idx]; y = [y[i] for i in idx]
    split = int(len(y) * 0.8)
    tree = build_tree(X[:split], y[:split], max_depth=5)
    print_tree(tree, ["x1", "x2"])
    correct = sum(1 for i in range(split, len(y)) if predict(tree, X[i]) == y[i])
    print(f"\nAccuracy: {correct}/{len(y)-split} = {correct/(len(y)-split)*100:.1f}%")

def main():
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        with open(sys.argv[1]) as f:
            reader = csv.reader(f); headers = next(reader)
            data = list(reader)
        X = [[float(v) for v in row[:-1]] for row in data]
        y = [row[-1] for row in data]
        tree = build_tree(X, y)
        print_tree(tree, headers[:-1])
        correct = sum(1 for i in range(len(y)) if predict(tree, X[i]) == y[i])
        print(f"\nTraining accuracy: {correct}/{len(y)} = {correct/len(y)*100:.1f}%")
    else: demo()

if __name__ == "__main__": main()
