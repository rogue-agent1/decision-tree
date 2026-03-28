#!/usr/bin/env python3
"""decision_tree - Simple decision tree classifier (ID3-like)."""
import sys, math, collections, json

def entropy(labels):
    n=len(labels); counts=collections.Counter(labels)
    return -sum(c/n*math.log2(c/n) for c in counts.values() if c>0)

def info_gain(data, labels, feature_idx):
    total_ent=entropy(labels); n=len(data)
    values=set(row[feature_idx] for row in data)
    weighted=0
    for v in values:
        subset_labels=[labels[i] for i,row in enumerate(data) if row[feature_idx]==v]
        weighted+=len(subset_labels)/n*entropy(subset_labels)
    return total_ent-weighted

def build_tree(data, labels, features, depth=0, max_depth=10):
    if len(set(labels))==1: return labels[0]
    if not features or depth>=max_depth: return collections.Counter(labels).most_common(1)[0][0]
    gains=[(info_gain(data,labels,f),f) for f in features]
    best_f=max(gains,key=lambda x:x[0])[1]
    tree={'feature':best_f,'children':{}}
    values=set(row[best_f] for row in data)
    remaining=[f for f in features if f!=best_f]
    for v in values:
        sub_data=[row for row in data if row[best_f]==v]
        sub_labels=[labels[i] for i,row in enumerate(data) if row[best_f]==v]
        tree['children'][v]=build_tree(sub_data,sub_labels,remaining,depth+1,max_depth)
    return tree

def predict(tree, sample):
    if not isinstance(tree,dict): return tree
    val=sample[tree['feature']]
    if val in tree['children']: return predict(tree['children'][val],sample)
    return list(tree['children'].values())[0] if tree['children'] else '?'

def display(tree, indent=0):
    if not isinstance(tree,dict): print(f"{'  '*indent}-> {tree}"); return
    for val,subtree in tree['children'].items():
        print(f"{'  '*indent}[feature {tree['feature']} = {val}]")
        display(subtree,indent+1)

def demo():
    # Play tennis dataset
    data=[['Sunny','Hot','High','Weak'],['Sunny','Hot','High','Strong'],
          ['Overcast','Hot','High','Weak'],['Rain','Mild','High','Weak'],
          ['Rain','Cool','Normal','Weak'],['Rain','Cool','Normal','Strong'],
          ['Overcast','Cool','Normal','Strong'],['Sunny','Mild','High','Weak'],
          ['Sunny','Cool','Normal','Weak'],['Rain','Mild','Normal','Weak'],
          ['Sunny','Mild','Normal','Strong'],['Overcast','Mild','High','Strong'],
          ['Overcast','Hot','Normal','Weak'],['Rain','Mild','High','Strong']]
    labels=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
    tree=build_tree(data,labels,[0,1,2,3])
    print("Decision Tree (Play Tennis):")
    display(tree)
    test=['Sunny','Cool','High','Strong']
    print(f"\n  Predict {test}: {predict(tree,test)}")

def main():
    args=sys.argv[1:]
    if not args or args[0]=='demo': demo()

if __name__=='__main__': main()
