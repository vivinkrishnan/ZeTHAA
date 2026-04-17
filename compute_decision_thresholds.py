import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

df=pd.read_csv("synthetic_authentication_dataset.csv")

labels=(df["Label"]=="Attack").astype(int)
risk=df["Risk_Score"]

fpr,tpr,thr=roc_curve(labels,risk)

valid = np.isfinite(thr)

fpr = fpr[valid]
tpr = tpr[valid]
thr = thr[valid]

youden = tpr-fpr
best_index=np.argmax(youden)
best=thr[best_index]

benign=df[df["Label"]=="Benign"]["Risk_Score"]
step=benign.quantile(0.75)

with open("risk_policy_thresholds.txt","w") as f:
    f.write(f"{step:.6f},{best:.6f}")
    
print("Decision thresholds computed:")
print("Step-up threshold:", step)
print("Block threshold:", best)