import subprocess
import pandas as pd
import numpy as np

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score
)

attack_levels = [0.1,0.2,0.3,0.4]

results=[]

for level in attack_levels:

    print("\nRunning experiment with attack ratio:", level)

    # generate dataset
    subprocess.run(["python","zt_dataset_generator.py",str(level)])

    # compute weights
    subprocess.run(["python","compute_risk_weights.py"])

    # compute thresholds
    subprocess.run(["python","compute_decision_thresholds.py"])

    df = pd.read_csv("synthetic_authentication_dataset.csv")

    labels = (df["Label"]=="Attack").astype(int)

    with open("risk_policy_thresholds.txt") as f:
        tau1,tau2 = map(float,f.read().split(","))

    pred = df["Risk_Score"] >= tau2

    accuracy = accuracy_score(labels,pred)
    precision = precision_score(labels,pred)
    recall = recall_score(labels,pred)
    f1 = f1_score(labels,pred)
    auc = roc_auc_score(labels,df["Risk_Score"])

    results.append({
    "Attack_Ratio":level,
    "Accuracy":accuracy,
    "Precision":precision,
    "Recall":recall,
    "F1":f1,
    "AUC":auc
    })

results_df=pd.DataFrame(results)

results_df.to_csv("attack_intensity_results.csv",index=False)

print("\nExperiment complete\n")
print(results_df)