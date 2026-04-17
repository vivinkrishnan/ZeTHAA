import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("synthetic_authentication_dataset.csv")

# Label
df["Attack_Flag"] = (df["Label"] == "Attack").astype(int)

# -----------------------------
# Load thresholds (τ1, τ2)
# -----------------------------
with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))

# -----------------------------
# Signals
# -----------------------------
signals = [
    "Geo_Anomaly",
    "Travel_Anomaly",
    "Timezone_Anomaly",
    "Temporal_Anomaly",
    "Fingerprint_Mismatch",
    "Token_Context_Mismatch",
    "Repeated_Attacker_IP"
]

# -----------------------------
# Load calibrated weights
# -----------------------------
weights_df = pd.read_csv("calibrated_weights.csv")
weight_map = dict(zip(weights_df["Feature"], weights_df["Weight"]))


# -----------------------------
# Derive contextual signals
# -----------------------------
df["Geo_Anomaly"] = (df["Geo_Location"] != df.groupby("User_ID")["Geo_Location"].transform("first"))

df["Travel_Anomaly"] = df["Travel_Status"].astype(bool)

df["Timezone_Anomaly"] = (df["Timezone_Shift"] != 0)

df["Temporal_Anomaly"] = (df["Login_Hour"] < 6) | (df["Login_Hour"] > 22)

df["Fingerprint_Mismatch"] = df["User_Agent"].str.contains("Python|curl|bot", case=False, na=False)

df["Token_Context_Mismatch"] = (
    df.groupby("Session_Token_ID")["Geo_Location"].transform("nunique") > 1
)

df["Repeated_Attacker_IP"] = (
    df.groupby("IP_Address")["Attack_Flag"].transform("sum") > 3
)

# -----------------------------
# Risk recomputation (CORRECT)
# -----------------------------
def compute_risk(row, active_signals):
    """
    Start from original Risk_Score and REMOVE contribution
    of signals not in active_signals.
    """
    risk = row["Risk_Score"]

    for s in signals:
        if s not in row:
            continue
        if s not in active_signals and row[s]:
            risk -= weight_map.get(s, 0)

    return max(0, risk)


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(risk_scores, labels):
    preds = (risk_scores >= tau2).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, risk_scores)

    return acc, prec, rec, f1, auc


# -----------------------------
# FULL MODEL (baseline)
# -----------------------------
results = []

full_risk = df["Risk_Score"]
acc, prec, rec, f1, auc = evaluate(full_risk, df["Attack_Flag"])

results.append({
    "Removed_Signal": "None (Full Model)",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "AUC": auc
})

# -----------------------------
# SINGLE SIGNAL ABLATION
# -----------------------------
for s in signals:
    active = [sig for sig in signals if sig != s]

    df["ablated_risk"] = df.apply(lambda r: compute_risk(r, active), axis=1)

    acc, prec, rec, f1, auc = evaluate(df["ablated_risk"], df["Attack_Flag"])

    results.append({
        "Removed_Signal": s,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    })

single_df = pd.DataFrame(results)

print("\nSingle Signal Ablation Results:\n")
print(single_df)


# -----------------------------
# GROUPED ABLATION
# -----------------------------
groups = {
    "Mobility": ["Geo_Anomaly", "Travel_Anomaly"],
    "Temporal": ["Timezone_Anomaly", "Temporal_Anomaly"],
    "Device": ["Fingerprint_Mismatch"],
    "Session": ["Token_Context_Mismatch"],
    "Campaign": ["Repeated_Attacker_IP"]
}

group_results = []

for group, group_signals in groups.items():
    active = [s for s in signals if s not in group_signals]

    df["ablated_risk"] = df.apply(lambda r: compute_risk(r, active), axis=1)

    acc, prec, rec, f1, auc = evaluate(df["ablated_risk"], df["Attack_Flag"])

    group_results.append({
        "Removed_Group": group,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    })

group_df = pd.DataFrame(group_results)

print("\nGrouped Signal Ablation Results:\n")
print(group_df)