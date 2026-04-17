import pandas as pd

df=pd.read_csv("synthetic_authentication_dataset.csv")

with open("risk_policy_thresholds.txt") as f:
    tau1,tau2=map(float,f.read().split(","))

def decision(r):
    if r<tau1:return"ALLOW"
    elif r<tau2:return"STEP_UP"
    else:return"BLOCK"

df["Decision"]=df["Risk_Score"].apply(decision)

print(df.groupby(["Decision","Label"]).size().unstack())