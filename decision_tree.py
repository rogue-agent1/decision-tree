#!/usr/bin/env python3
"""decision_tree - Decision tree classifier."""
import argparse, math, sys, json

def entropy(labels):
    n = len(labels)
    if n == 0: return 0
    counts = {}
    for l in labels: counts[l] = counts.get(l, 0) + 1
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c > 0)

def info_gain(data, labels, feat_idx, threshold):
    left_l, right_l = [], []
    for i, row in enumerate(data):
        if row[feat_idx] <= threshold: left_l.append(labels[i])
        else: right_l.append(labels[i])
    n = len(labels)
    if not left_l or not right_l: return 0
    return entropy(labels) - (len(left_l)/n)*entropy(left_l) - (len(right_l)/n)*entropy(right_l)

def best_split(data, labels):
    best_gain, best_feat, best_thresh = 0, 0, 0
    for f in range(len(data[0])):
        vals = sorted(set(row[f] for row in data))
        for i in range(len(vals)-1):
            thresh = (vals[i]+vals[i+1])/2
            gain = info_gain(data, labels, f, thresh)
            if gain > best_gain: best_gain, best_feat, best_thresh = gain, f, thresh
    return best_feat, best_thresh, best_gain

def build_tree(data, labels, depth=0, max_depth=10):
    if len(set(labels)) == 1: return {"leaf": labels[0]}
    if depth >= max_depth or len(data) < 2:
        counts = {}
        for l in labels: counts[l] = counts.get(l,0)+1
        return {"leaf": max(counts, key=counts.get)}
    feat, thresh, gain = best_split(data, labels)
    if gain == 0:
        counts = {}
        for l in labels: counts[l] = counts.get(l,0)+1
        return {"leaf": max(counts, key=counts.get)}
    left_d, left_l, right_d, right_l = [], [], [], []
    for i, row in enumerate(data):
        if row[feat] <= thresh: left_d.append(row); left_l.append(labels[i])
        else: right_d.append(row); right_l.append(labels[i])
    return {"feat":feat,"thresh":thresh,
            "left":build_tree(left_d,left_l,depth+1,max_depth),
            "right":build_tree(right_d,right_l,depth+1,max_depth)}

def predict(tree, row):
    if "leaf" in tree: return tree["leaf"]
    if row[tree["feat"]] <= tree["thresh"]: return predict(tree["left"], row)
    return predict(tree["right"], row)

def print_tree(tree, indent=0):
    if "leaf" in tree: print(" "*indent + f"-> {tree['leaf']}"); return
    print(" "*indent + f"feat[{tree['feat']}] <= {tree['thresh']:.2f}?")
    print_tree(tree["left"], indent+2)
    print_tree(tree["right"], indent+2)

def main():
    p = argparse.ArgumentParser(description="Decision tree classifier")
    p.add_argument("--demo", choices=["iris","xor"], default="iris")
    p.add_argument("-d","--depth", type=int, default=5)
    a = p.parse_args()
    if a.demo == "xor":
        data = [[0,0],[0,1],[1,0],[1,1]]; labels = [0,1,1,0]
    else:  # simple iris-like
        import random; random.seed(42)
        data = []; labels = []
        for _ in range(50): data.append([random.gauss(5,0.5),random.gauss(3.5,0.3)]); labels.append("setosa")
        for _ in range(50): data.append([random.gauss(6,0.5),random.gauss(2.8,0.3)]); labels.append("versicolor")
        for _ in range(50): data.append([random.gauss(7,0.5),random.gauss(3.2,0.3)]); labels.append("virginica")
    tree = build_tree(data, labels, max_depth=a.depth)
    print("Tree:"); print_tree(tree)
    correct = sum(predict(tree,row)==label for row,label in zip(data,labels))
    print(f"\nAccuracy: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")

if __name__ == "__main__": main()
