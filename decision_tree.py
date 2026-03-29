#!/usr/bin/env python3
"""Decision tree — ID3/CART-style classifier."""
import math, sys
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth; self.min_samples = min_samples; self.tree = None
    def _entropy(self, labels):
        n = len(labels); counts = Counter(labels)
        return -sum((c/n) * math.log2(c/n) for c in counts.values() if c > 0)
    def _best_split(self, X, y):
        best_gain = 0; best_feat = None; best_val = None
        parent_entropy = self._entropy(y); n = len(y)
        for feat in range(len(X[0])):
            values = sorted(set(row[feat] for row in X))
            for val in values:
                left_y = [y[i] for i in range(n) if X[i][feat] <= val]
                right_y = [y[i] for i in range(n) if X[i][feat] > val]
                if not left_y or not right_y: continue
                gain = parent_entropy - (len(left_y)/n * self._entropy(left_y) + len(right_y)/n * self._entropy(right_y))
                if gain > best_gain: best_gain = gain; best_feat = feat; best_val = val
        return best_feat, best_val, best_gain
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        feat, val, gain = self._best_split(X, y)
        if gain == 0: return Counter(y).most_common(1)[0][0]
        left_idx = [i for i in range(len(X)) if X[i][feat] <= val]
        right_idx = [i for i in range(len(X)) if X[i][feat] > val]
        return {"feat": feat, "val": val,
                "left": self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth+1),
                "right": self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth+1)}
    def fit(self, X, y): self.tree = self._build(X, y, 0)
    def _predict_one(self, x, node):
        if not isinstance(node, dict): return node
        if x[node["feat"]] <= node["val"]: return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])
    def predict(self, X): return [self._predict_one(x, self.tree) for x in X]

if __name__ == "__main__":
    import random; random.seed(42)
    X, y = [], []
    for _ in range(100):
        x1, x2 = random.random()*10, random.random()*10
        X.append([x1, x2]); y.append("A" if x1 + x2 > 10 else "B")
    dt = DecisionTree(max_depth=5); dt.fit(X[:80], y[:80])
    preds = dt.predict(X[80:])
    acc = sum(1 for p, a in zip(preds, y[80:]) if p == a) / 20
    print(f"Decision Tree: accuracy={acc*100:.0f}%")
    print(f"Predict [8,8]: {dt._predict_one([8,8], dt.tree)}")
    print(f"Predict [2,2]: {dt._predict_one([2,2], dt.tree)}")
