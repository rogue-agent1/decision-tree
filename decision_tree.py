#!/usr/bin/env python3
"""Decision tree classifier (ID3/CART)."""
import math
from collections import Counter
def entropy(labels):
    n=len(labels);counts=Counter(labels)
    return -sum(c/n*math.log2(c/n) for c in counts.values() if c>0)
def gini(labels):
    n=len(labels);counts=Counter(labels)
    return 1-sum((c/n)**2 for c in counts.values())
def info_gain(data,labels,feature_idx,criterion="entropy"):
    fn=entropy if criterion=="entropy" else gini
    parent=fn(labels);n=len(data)
    values=set(row[feature_idx] for row in data)
    child=0
    for v in values:
        subset=[labels[i] for i in range(n) if data[i][feature_idx]==v]
        child+=len(subset)/n*fn(subset)
    return parent-child
class DecisionTree:
    def __init__(self,max_depth=10,criterion="entropy"):
        self.max_depth=max_depth;self.criterion=criterion;self.tree=None
    def fit(self,data,labels):
        features=list(range(len(data[0])))
        self.tree=self._build(data,labels,features,0)
    def _build(self,data,labels,features,depth):
        if len(set(labels))==1: return {"leaf":True,"label":labels[0]}
        if depth>=self.max_depth or not features: return {"leaf":True,"label":Counter(labels).most_common(1)[0][0]}
        gains=[(info_gain(data,labels,f,self.criterion),f) for f in features]
        best_gain,best_f=max(gains)
        if best_gain<=0: return {"leaf":True,"label":Counter(labels).most_common(1)[0][0]}
        values=set(row[best_f] for row in data)
        children={}
        for v in values:
            idx=[i for i in range(len(data)) if data[i][best_f]==v]
            sub_data=[data[i] for i in idx];sub_labels=[labels[i] for i in idx]
            remaining=[f for f in features if f!=best_f]
            children[v]=self._build(sub_data,sub_labels,remaining,depth+1)
        return {"leaf":False,"feature":best_f,"children":children,"default":Counter(labels).most_common(1)[0][0]}
    def predict(self,row):
        node=self.tree
        while not node["leaf"]:
            val=row[node["feature"]]
            if val in node["children"]: node=node["children"][val]
            else: return node["default"]
        return node["label"]
if __name__=="__main__":
    data=[[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1]]
    labels=[0,1,1,0,0,1,1,0]
    dt=DecisionTree();dt.fit(data,labels)
    correct=sum(1 for x,y in zip(data,labels) if dt.predict(x)==y)
    print(f"Accuracy: {correct}/{len(data)}")
    print("Decision tree OK")
