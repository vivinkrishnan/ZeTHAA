import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

df=pd.read_csv("synthetic_authentication_dataset.csv")
with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))


actual = df["Label"]=="Attack"

risk_pred = df["Risk_Score"]>tau2
lr_pred = df["LR_Prediction"]==1

n01 = sum((risk_pred == actual) & (lr_pred != actual))
n10 = sum((risk_pred != actual) & (lr_pred == actual))

table=[[0,n01],[n10,0]]

result = mcnemar(table, exact=True)

print(result)