#!/usr/bin/env python3
"""Decision tree classifier from scratch."""
import sys, math, json, csv
from collections import Counter

def entropy(labels):
    n = len(labels); counts = Counter(labels)
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c)

def info_gain(data, labels, feature_idx):
    parent_ent = entropy(labels)
    values = set(row[feature_idx] for row in data)
    child_ent = 0
    for v in values:
        subset = [labels[i] for i, row in enumerate(data) if row[feature_idx] == v]
        child_ent += len(subset)/len(labels) * entropy(subset)
    return parent_ent - child_ent

class Node:
    def __init__(self, feature=None, value=None, label=None, children=None):
        self.feature = feature; self.value = value
        self.label = label; self.children = children or {}

def build_tree(data, labels, features, depth=0, max_depth=10):
    if len(set(labels)) == 1: return Node(label=labels[0])
    if not features or depth >= max_depth: return Node(label=Counter(labels).most_common(1)[0][0])
    gains = [(info_gain(data, labels, f), f) for f in features]
    best_gain, best_feat = max(gains)
    if best_gain <= 0: return Node(label=Counter(labels).most_common(1)[0][0])
    node = Node(feature=best_feat)
    values = set(row[best_feat] for row in data)
    remaining = [f for f in features if f != best_feat]
    for v in values:
        subset_idx = [i for i, row in enumerate(data) if row[best_feat] == v]
        sub_data = [data[i] for i in subset_idx]
        sub_labels = [labels[i] for i in subset_idx]
        node.children[v] = build_tree(sub_data, sub_labels, remaining, depth+1, max_depth)
    return node

def predict(node, sample):
    if node.label is not None: return node.label
    val = sample[node.feature]
    if val in node.children: return predict(node.children[val], sample)
    return node.label or 'unknown'

def print_tree(node, indent=0, names=None):
    if node.label is not None: print(' '*indent + f'→ {node.label}'); return
    fname = names[node.feature] if names else f'feature[{node.feature}]'
    for v, child in sorted(node.children.items(), key=str):
        print(' '*indent + f'{fname} = {v}:')
        print_tree(child, indent+2, names)

if __name__ == '__main__':
    if '--demo' in sys.argv:
        # Classic: play tennis?
        data = [['sunny','hot','high','weak'],['sunny','hot','high','strong'],['overcast','hot','high','weak'],
                ['rain','mild','high','weak'],['rain','cool','normal','weak'],['rain','cool','normal','strong'],
                ['overcast','cool','normal','strong'],['sunny','mild','high','weak'],['sunny','cool','normal','weak'],
                ['rain','mild','normal','weak'],['sunny','mild','normal','strong'],['overcast','mild','high','strong'],
                ['overcast','hot','normal','weak'],['rain','mild','high','strong']]
        labels = ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']
        names = ['outlook','temp','humidity','wind']
        tree = build_tree(data, labels, list(range(4)))
        print("Decision Tree (Play Tennis?):\n")
        print_tree(tree, names=names)
        print(f"\nPredict sunny/cool/normal/weak: {predict(tree, ['sunny','cool','normal','weak'])}")
    else:
        print("Usage: decision_tree.py --demo")
