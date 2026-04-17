import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("synthetic_authentication_dataset.csv")

labels = (df["Label"] == "Attack").astype(int)
risk = df["Risk_Score"]

# --------------------------------------------------
# Load policy thresholds
# --------------------------------------------------

with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))

threshold = tau2

# --------------------------------------------------
# Predictions for proposed framework
# --------------------------------------------------

pred = risk >= threshold

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(labels, pred)

print("\nConfusion Matrix")
print(cm)

# --------------------------------------------------
# Performance Metrics
# --------------------------------------------------

accuracy = accuracy_score(labels, pred)
precision = precision_score(labels, pred)
recall = recall_score(labels, pred)
f1 = f1_score(labels, pred)

print("\nPerformance Metrics")
print("Accuracy:", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall:", round(recall,4))
print("F1 Score:", round(f1,4))

# --------------------------------------------------
# ROC and AUC for proposed framework
# --------------------------------------------------

fpr_fw, tpr_fw, _ = roc_curve(labels, risk)
auc_fw = auc(fpr_fw, tpr_fw)

print("\nAUC (Proposed Framework):", round(auc_fw,4))

# --------------------------------------------------
# Risk score distribution
# --------------------------------------------------

plt.figure()

sns.histplot(
    df[df["Label"]=="Benign"]["Risk_Score"],
    bins=30,
    color="blue",
    label="Benign",
    alpha=0.5
)

sns.histplot(
    df[df["Label"]=="Attack"]["Risk_Score"],
    bins=30,
    color="red",
    label="Attack",
    alpha=0.5
)

plt.legend()
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")

plt.savefig("risk_distribution.png")

# --------------------------------------------------
# Feature engineering for baseline models
# --------------------------------------------------

df["Geo_Anomaly"] = df.groupby("User_ID")["Geo_Location"].transform(
    lambda x: x != x.mode()[0]
)

df["Fingerprint_Mismatch"] = df.groupby("Session_ID")["User_Agent"].transform(
    "nunique"
) > 1

df["Token_Context_Mismatch"] = df.groupby("Session_Token_ID")[
    "Geo_Location"
].transform("nunique") > 1

df["Repeated_Attacker_IP"] = df.groupby("IP_Address")[
    "Label"
].transform(lambda x: (x=="Attack").sum()) > 3
df["Timezone_Anomaly"] = abs(df["Timezone_Shift"]) > 3
df["Travel_Anomaly"] = df["Travel_Status"]

# df["Login_Hour"] = df["Time_of_Access"].dt.hour
df["Login_Hour"] = pd.to_datetime(df["Time_of_Access"]).dt.hour

df["User_Mean_Login"] = df.groupby("User_ID")["Login_Hour"].transform("mean")
df["Temporal_Anomaly"] = abs(df["Login_Hour"] - df["User_Mean_Login"]) >4

# df["Temporal_Anomaly"] = (
#     (df["Is_Weekend"] == True) &
#     (df["Login_Hour"] < 6)
# )

features = [
"Geo_Anomaly",
"Travel_Anomaly",
"Timezone_Anomaly",
"Temporal_Anomaly",
"Fingerprint_Mismatch",
"Token_Context_Mismatch",
"Repeated_Attacker_IP"
]

X = df[features].astype(int)
y = labels

# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# --------------------------------------------------
# Logistic Regression baseline
# --------------------------------------------------

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_probs = lr.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
auc_lr = auc(fpr_lr, tpr_lr)

print("AUC (Logistic Regression):", round(auc_lr,4))

# --------------------------------------------------
# Random Forest baseline
# --------------------------------------------------

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:,1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_rf = auc(fpr_rf, tpr_rf)

print("AUC (Random Forest):", round(auc_rf,4))

# --------------------------------------------------
# ROC comparison plot
# --------------------------------------------------

plt.figure()

plt.plot(fpr_fw, tpr_fw, color='blue',lw=2, label=f"Proposed (AUC={auc_fw:.3f})")
plt.plot(fpr_lr, tpr_lr, color='orange',lw=4, label=f"Logistic Regression (AUC={auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, color='green',lw=2, linestyle="--",  label=f"Random Forest (AUC={auc_rf:.3f})")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Model Comparison ROC")

plt.legend()

plt.savefig("model_comparison_roc.png")

print("\nEvaluation complete.")

actual = df["Label"]=="Attack"

df["LR_Prediction"] = lr.predict(X)


risk_pred = df["Risk_Score"]>tau2
lr_pred = df["LR_Prediction"]==1

n01 = sum((risk_pred == actual) & (lr_pred != actual))
n10 = sum((risk_pred != actual) & (lr_pred == actual))

table=[[0,n01],[n10,0]]

result = mcnemar(table, exact=True)

print(result)