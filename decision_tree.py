#!/usr/bin/env python3
"""Decision tree classifier (ID3)."""
import sys, math, collections
def entropy(labels):
    n=len(labels); counts=collections.Counter(labels)
    return -sum(c/n*math.log2(c/n) for c in counts.values() if c>0)
def best_split(data,labels,features):
    best_ig,best_f=-1,None; base=entropy(labels)
    for f in features:
        vals=set(d[f] for d in data); ig=base
        for v in vals:
            sub=[labels[i] for i,d in enumerate(data) if d[f]==v]
            ig-=len(sub)/len(labels)*entropy(sub)
        if ig>best_ig: best_ig,best_f=ig,f
    return best_f
def build(data,labels,features):
    if len(set(labels))==1: return labels[0]
    if not features: return collections.Counter(labels).most_common(1)[0][0]
    f=best_split(data,labels,features)
    if f is None: return collections.Counter(labels).most_common(1)[0][0]
    tree={'feature':f,'children':{}}
    for v in set(d[f] for d in data):
        idx=[i for i,d in enumerate(data) if d[f]==v]
        tree['children'][v]=build([data[i] for i in idx],[labels[i] for i in idx],[x for x in features if x!=f])
    return tree
def predict(tree,sample):
    if isinstance(tree,str): return tree
    v=sample.get(tree['feature']); return predict(tree['children'].get(v,list(tree['children'].values())[0]),sample)
# Demo
data=[{'outlook':'sunny','temp':'hot','play':'no'},{'outlook':'sunny','temp':'mild','play':'yes'},
      {'outlook':'overcast','temp':'hot','play':'yes'},{'outlook':'rain','temp':'mild','play':'yes'},
      {'outlook':'rain','temp':'cool','play':'no'}]
labels=[d.pop('play') for d in data]
tree=build(data,labels,['outlook','temp'])
print(f"Tree: {tree}")
for d in [{'outlook':'sunny','temp':'mild'},{'outlook':'rain','temp':'cool'}]:
    print(f"  {d} → {predict(tree,d)}")
