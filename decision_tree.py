#!/usr/bin/env python3
"""decision_tree - Decision tree classifier."""
import sys,argparse,json,math,random
from collections import Counter
def entropy(labels):
    n=len(labels);freq=Counter(labels)
    return -sum((c/n)*math.log2(c/n) for c in freq.values() if c>0)
def best_split(X,y,features):
    best_gain=-1;best_feat=0;best_val=0
    base_ent=entropy(y)
    for f in features:
        vals=sorted(set(row[f] for row in X))
        for v in vals:
            left_y=[yi for xi,yi in zip(X,y) if xi[f]<=v]
            right_y=[yi for xi,yi in zip(X,y) if xi[f]>v]
            if not left_y or not right_y:continue
            gain=base_ent-(len(left_y)/len(y)*entropy(left_y)+len(right_y)/len(y)*entropy(right_y))
            if gain>best_gain:best_gain=gain;best_feat=f;best_val=v
    return best_feat,best_val,best_gain
def build_tree(X,y,depth=0,max_depth=5):
    if depth>=max_depth or len(set(y))==1:return Counter(y).most_common(1)[0][0]
    feat,val,gain=best_split(X,y,range(len(X[0])))
    if gain<=0:return Counter(y).most_common(1)[0][0]
    left_X,left_y,right_X,right_y=[],[],[],[]
    for xi,yi in zip(X,y):
        if xi[feat]<=val:left_X.append(xi);left_y.append(yi)
        else:right_X.append(xi);right_y.append(yi)
    return {"feature":feat,"threshold":round(val,3),"left":build_tree(left_X,left_y,depth+1,max_depth),"right":build_tree(right_X,right_y,depth+1,max_depth)}
def predict(tree,x):
    if not isinstance(tree,dict):return tree
    if x[tree["feature"]]<=tree["threshold"]:return predict(tree["left"],x)
    return predict(tree["right"],x)
def main():
    p=argparse.ArgumentParser(description="Decision tree")
    p.add_argument("--max-depth",type=int,default=5);p.add_argument("--samples",type=int,default=100)
    args=p.parse_args()
    random.seed(42)
    X=[];y=[]
    for _ in range(args.samples//2):
        X.append([random.gauss(2,1),random.gauss(2,1)]);y.append("A")
        X.append([random.gauss(5,1),random.gauss(5,1)]);y.append("B")
    split=int(len(X)*0.8)
    tree=build_tree(X[:split],y[:split],max_depth=args.max_depth)
    correct=sum(1 for xi,yi in zip(X[split:],y[split:]) if predict(tree,xi)==yi)
    acc=correct/len(X[split:]) if len(X)>split else 0
    print(json.dumps({"tree":tree,"accuracy":round(acc,4),"test_size":len(X)-split},indent=2))
if __name__=="__main__":main()
