import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

# ----------------------------------------------------
# Load dataset
# ----------------------------------------------------

df = pd.read_csv("synthetic_authentication_dataset.csv")

df["Attack_Flag"] = (df["Label"]=="Attack").astype(int)

# ----------------------------------------------------
# Load thresholds
# ----------------------------------------------------

with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))

# ----------------------------------------------------
# TABLE 1 — Dataset Summary
# ----------------------------------------------------

table1 = pd.DataFrame({
"Metric":[
"Users",
"Sessions",
"Events",
"Attack Events",
"Attack Ratio"
],
"Value":[
df["User_ID"].nunique(),
df["Session_ID"].nunique(),
len(df),
df["Attack_Flag"].sum(),
df["Attack_Flag"].mean()
]
})

print("\nTABLE 1 – DATASET SUMMARY\n")
print(table1)

# ----------------------------------------------------
# TABLE 2 — Attack Distribution
# ----------------------------------------------------

table2 = df["Attack_Type"].value_counts().reset_index()
table2.columns = ["Attack Type","Count"]

print("\nTABLE 2 – ATTACK DISTRIBUTION\n")
print(table2)

# ----------------------------------------------------
# TABLE 3 — Mobility Statistics
# ----------------------------------------------------

table3 = pd.DataFrame({
"Metric":[
"Travel Sessions",
"Home Sessions"
],
"Value":[
df["Travel_Status"].sum(),
(~df["Travel_Status"]).sum()
]
})

print("\nTABLE 3 – USER MOBILITY\n")
print(table3)

# ----------------------------------------------------
# TABLE 4 — Time Behavior
# ----------------------------------------------------

table4 = df.groupby("Weekday").size().reset_index()
table4.columns = ["Weekday","Login Count"]

print("\nTABLE 4 – LOGIN ACTIVITY BY WEEKDAY\n")
print(table4)

# ----------------------------------------------------
# TABLE 5 — Device Lifecycle
# ----------------------------------------------------

table5 = df.groupby("Device_Model")["User_ID"].nunique().reset_index()
table5.columns = ["Device Model","Users"]

print("\nTABLE 5 – DEVICE DISTRIBUTION\n")
print(table5)

# ----------------------------------------------------
# TABLE 6 — OS Version Distribution
# ----------------------------------------------------

table6 = df["OS_Version"].value_counts().reset_index()
table6.columns = ["OS Version","Count"]

print("\nTABLE 6 – OS VERSION DISTRIBUTION\n")
print(table6)

# ----------------------------------------------------
# TABLE 7 — Application Version Rollout
# ----------------------------------------------------

table7 = df["Application_Version"].value_counts().reset_index()
table7.columns = ["Application Version","Count"]

print("\nTABLE 7 – APP VERSION DISTRIBUTION\n")
print(table7)

# ----------------------------------------------------
# TABLE 8 — Risk Score Statistics
# ----------------------------------------------------

table8 = pd.DataFrame({
"Metric":[
"Mean Risk (Benign)",
"Mean Risk (Attack)"
],
"Value":[
df[df["Label"]=="Benign"]["Risk_Score"].mean(),
df[df["Label"]=="Attack"]["Risk_Score"].mean()
]
})

print("\nTABLE 8 – RISK SCORE STATISTICS\n")
print(table8)

# ----------------------------------------------------
# TABLE 9 — Confusion Matrix
# ----------------------------------------------------

pred = df["Risk_Score"] >= tau2

cm = confusion_matrix(df["Attack_Flag"], pred)

table9 = pd.DataFrame(cm,
index=["Actual Benign","Actual Attack"],
columns=["Predicted Benign","Predicted Attack"])

print("\nTABLE 9 – CONFUSION MATRIX\n")
print(table9)

# ----------------------------------------------------
# TABLE 10 — Model Performance
# ----------------------------------------------------

accuracy = accuracy_score(df["Attack_Flag"], pred)
precision = precision_score(df["Attack_Flag"], pred)
recall = recall_score(df["Attack_Flag"], pred)
f1 = f1_score(df["Attack_Flag"], pred)
auc = roc_auc_score(df["Attack_Flag"], df["Risk_Score"])

table10 = pd.DataFrame({
"Metric":["Accuracy","Precision","Recall","F1 Score","AUC"],
"Value":[accuracy,precision,recall,f1,auc]
})

print("\nTABLE 10 – MODEL PERFORMANCE\n")
print(table10)

# ----------------------------------------------------
# TABLE 11 — Authentication Policy Outcomes
# ----------------------------------------------------

def decision(r):

    if r < tau1:
        return "ALLOW"
    elif r < tau2:
        return "STEP_UP"
    else:
        return "BLOCK"

df["Decision"] = df["Risk_Score"].apply(decision)

table11 = df.groupby(["Decision","Label"]).size().unstack()

print("\nTABLE 11 – POLICY DECISIONS\n")
print(table11)

# ----------------------------------------------------
# TABLE 12 — User Friction
# ----------------------------------------------------

benign = df[df["Label"]=="Benign"]

stepup_rate = (benign["Decision"]=="STEP_UP").mean()
false_block_rate = (benign["Decision"]=="BLOCK").mean()

table12 = pd.DataFrame({
"Metric":["Step-Up Rate","False Block Rate"],
"Value":[stepup_rate,false_block_rate]
})

print("\nTABLE 12 – USER FRICTION\n")
print(table12)

print("\nAll results tables generated.")