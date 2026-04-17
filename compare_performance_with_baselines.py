import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,roc_curve

df=pd.read_csv("synthetic_authentication_dataset.csv")

with open("risk_policy_thresholds.txt") as f:
    tau1_actual,tau2_actual=map(float,f.read().split(","))
    
df["Attack_Flag"] = (df["Label"]=="Attack").astype(int)

def calibrate(prob):
    scale = tau2_actual/np.percentile(prob, 90)
    return np.clip(prob*scale, 0, 1)

def decision(r):
    if r<tau1_actual:
        return "ALLOW"
    elif r<tau2_actual:
        return "STEP_UP"
    else:
        return "BLOCK"

def heuristic_decision(row):
    score = 0
    
    if row["Fingerprint_Mismatch"]:
        score += 0.6
    if row["Token_Context_Mismatch"]:
        score += 0.5
    if row["Geo_Anomaly"]:
        score += 0.4
    if row["Repeated_Attacker_IP"]:
        score += 0.5
    if row["Travel_Anomaly"]:
        score += 0.1
    if row["Timezone_Anomaly"]:
        score += 0.1
    if row["Temporal_Anomaly"]:
        score += 0.1    
    
    return min(score, 1.0)

def compute_auc_safe(y_true, y_scores):
    try:
        if len(set(y_true)) < 2:    
            return 0.5
        else:
            return roc_auc_score(y_true, y_scores)
    except ValueError:
        return np.nan

def safe_roc(y_true, y_scores):
    if len(set(y_true)) < 2:
        return None, None,0.5
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, auc

def region_counts(scores):
    return pd.Series({
        "ALLOW": (scores < tau1_actual).mean(),
        "STEP_UP": ((scores >= tau1_actual) & (scores < tau2_actual)).mean(),
        "BLOCK": (scores >= tau2_actual).mean()
    })
    
def region_confusion_matrix(df,region_col):
    results = {}
    
    for region in ["ALLOW", "STEP_UP", "BLOCK"]:
        sub = df[df[region_col] == region]

        if len(sub) == 0:
            results[region] = {"Attack": 0, "Benign": 0}
        else:
            results[region] = {
                "Attack": (sub["Attack_Flag"]==1).sum(),
                "Benign": (sub["Attack_Flag"]==0).sum()
            }
    
    return pd.DataFrame(results).T
  
def region_confusion_matrix_normalized(df,region_col):
    out = {}
    
    for region in ["ALLOW", "STEP_UP", "BLOCK"]:
        sub = df[df[region_col] == region]

        if len(sub) == 0:
            out[region] = {"Attack": 0, "Benign": 0}
        else:
            total = len(sub)
            out[region] = {
                "Attack%": (sub["Attack_Flag"]==1).mean(),
                "Benign%": (sub["Attack_Flag"]==0).mean()
            }
    
    return pd.DataFrame(out).T

def compute_Dasu_heuristic(row):
    score = 0
    for k,w in weights.items():
        if k in row:
            score += w * row[k]
    return score/5
    
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx]+fnr[idx])/2
    eer_threshold = thresholds[idx]
    return eer,eer_threshold,fpr, tpr
    

    
cost_map={
    "ALLOW": 1,
    "STEP_UP": 5,
    "BLOCK": 10
}    

features = [
    "Geo_Anomaly",
    "Travel_Anomaly",
    "Timezone_Anomaly",
    "Temporal_Anomaly",
    "Fingerprint_Mismatch",
    "Token_Context_Mismatch",
    "Repeated_Attacker_IP"    
]

weights = {
    "Travel_Anomaly": 80/33,
    "Geo_Anomaly": 40/33,
    # "Temporal_Anomaly": 20/33,
    "Fingerprint_Mismatch": 20/33,
    "Token_Context_Mismatch": 5/33
}



df["Geo_Anomaly"]=df.groupby("User_ID")["Geo_Location"].transform(lambda x:x!=x.mode()[0])
df["Fingerprint_Mismatch"]=df.groupby("Session_ID")["User_Agent"].transform("nunique")>1
df["Token_Context_Mismatch"]=df.groupby("Session_Token_ID")["Geo_Location"].transform("nunique")>1
df["Repeated_Attacker_IP"]=df.groupby("IP_Address")["Attack_Flag"].transform("sum")>3
df["Timezone_Anomaly"] = abs(df["Timezone_Shift"]) > 3
df["Travel_Anomaly"] = df["Travel_Status"]
df["Temporal_Anomaly"] = (
    (df["Is_Weekend"] == True) &
    (df["Login_Hour"] < 6)
)

x = df[features].astype(int)
y = df["Attack_Flag"]

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(x, y,df, test_size=0.3, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

iso = IsolationForest(n_estimators=150,contamination=0.2, random_state=42)
iso_Matiushin = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_Matiushin.fit(X_train)
iso_test_Matiushin = iso_Matiushin.decision_function(X_test)
iso_train_Matiushin = iso_Matiushin.decision_function(X_train)

lof_Matiushin = LocalOutlierFactor(n_neighbors=35, contamination=0.01, novelty=True, metric='cosine')
lof_Matiushin.fit(X_train)
lof_test_Matiushin = lof_Matiushin.decision_function(X_test)
lof_train_Matiushin = lof_Matiushin.decision_function(X_train)

iso_train_Matiushin = (iso_train_Matiushin - iso_train_Matiushin.min()) / (iso_train_Matiushin.max() - iso_train_Matiushin.min())
iso_test_Matiushin = (iso_test_Matiushin - iso_test_Matiushin.min()) / (iso_test_Matiushin.max() - iso_test_Matiushin.min())
lof_train_Matiushin = (lof_train_Matiushin - lof_train_Matiushin.min()) / (lof_train_Matiushin.max() - lof_train_Matiushin.min())
lof_test_Matiushin = (lof_test_Matiushin - lof_test_Matiushin.min()) / (lof_test_Matiushin.max() - lof_test_Matiushin.min())

X_train_ml = X_train.copy()
X_test_ml = X_test.copy()

X_train_ml["Anomaly_Score_ISO_Matiushin"] = (iso_train_Matiushin + lof_train_Matiushin)/2
X_test_ml["Anomaly_Score_ISO_Matiushin"] = (iso_test_Matiushin + lof_test_Matiushin)/2

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=7,           # Increased depth for more complex splits
    reg_alpha=0.3,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: 1, 1: 2.5},  # Attack=1 is upweighted
    min_child_samples=20,
    random_state=42
)

model.fit(X_train_ml, y_train)
cal_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
cal_model.fit(X_train_ml, y_train)



lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gb.fit(X_train, y_train)
iso.fit(X_train)

df_test["LR_Prob"] = lr.predict_proba(X_test)[:,1]
df_test["RF_Prob"] = rf.predict_proba(X_test)[:,1]
df_test["XGB_Prob"] = xgb.predict_proba(X_test)[:,1]
df_test["GB_Prob"] = gb.predict_proba(X_test)[:,1]
df_test["Heuristic_Score"] = df_test.apply(heuristic_decision, axis=1)
iso_scores = iso.decision_function(X_test)
df_test["ISO_Score"] = -iso_scores
df_test["Dasu_Heuristic_Score"] = df_test.apply(compute_Dasu_heuristic, axis=1)
df_test["Matiushin_Prob"] = cal_model.predict_proba(X_test_ml)[:,1]


# print(np.mean(df_test["LR_Prob"]))
print(np.mean(df_test["RF_Prob"]))
print(np.mean(df_test["XGB_Prob"]))
print(np.mean(df_test["GB_Prob"]))
# print(np.mean(df_test["Heuristic_Score"]))
print(np.mean(df_test["RF_Prob"] == df_test["XGB_Prob"]))

df_test["LR_Cal"] = calibrate(df_test["LR_Prob"])
df_test["RF_Cal"] = calibrate(df_test["RF_Prob"])
df_test["XGB_Cal"] = calibrate(df_test["XGB_Prob"])
df_test["Heuristic_Cal"]= calibrate(df_test["Heuristic_Score"])
df_test["GB_Cal"] = calibrate(df_test["GB_Prob"])
df_test["ISO_Cal"] = calibrate(df_test["ISO_Score"])
df_test["Dasu_Heuristic_Cal"] = calibrate(df_test["Dasu_Heuristic_Score"])
df_test["Matiushin_Cal"] = calibrate(df_test["Matiushin_Prob"])


# print((np.mean((df_test["RF_Cal"] > tau2_actual) != (df_test["XGB_Cal"] > tau2_actual))))


df_test["Decision_Proposed"]=df_test["Risk_Score"].apply(decision)
df_test["Decision_LR"] = df_test["LR_Cal"].apply(decision)
df_test["Decision_RF"] = df_test["RF_Cal"].apply(decision)
df_test["Decision_XGB"] = df_test["XGB_Cal"].apply(decision)
df_test["Decision_Heuristic"] = df_test["Heuristic_Cal"].apply(decision)
df_test["Decision_GB"] = df_test["GB_Cal"].apply(decision)
df_test["Decision_ISO"] = df_test["ISO_Cal"].apply(decision)
df_test["Decision_Dasu_Heuristic"] = df_test["Dasu_Heuristic_Cal"].apply(decision)
df_test["Decision_Matiushin"] = df_test["Matiushin_Cal"].apply(decision)

print((df_test["Decision_RF"] == df_test["Decision_XGB"]).mean())
print(np.mean(df_test["Decision_RF"] == df_test["Decision_XGB"]))

def evaluate(decision_col):
    benign = df_test[df_test["Attack_Flag"]==0]
    attack = df_test[df_test["Attack_Flag"]==1]
    
    decisions = df_test[decision_col]
    
    
    return{
        "Accuracy": ((decisions == "BLOCK") == (df_test["Attack_Flag"] == 1) ).mean(),
        "Precision": (attack[decision_col] == "BLOCK").sum() / max((decisions == "BLOCK").sum(), 1),
        "Recall": (attack[decision_col] == "BLOCK").mean(),
        "F1-Score": 2 * (
            ((attack[decision_col] == "BLOCK").mean()) * 
            ((attack[decision_col] == "BLOCK").sum() / 
             max((decisions == "BLOCK").sum(), 1))
            ) / max(
                ((attack[decision_col] == "BLOCK").mean()) + 
                ((attack[decision_col] == "BLOCK").sum() / 
                 max((decisions == "BLOCK").sum(), 1)), 1e-6
        ),
        "Avg Cost": decisions.map(cost_map).mean(),
        "Step-Up Rate": (decisions == "STEP_UP").mean(),
        "Block Rate": (decisions == "BLOCK").mean(),
        "False Block Rate": (benign[decision_col]=="BLOCK").mean(),
        "Efficiency": (attack[decision_col] == "BLOCK").mean() / decisions.map(cost_map).mean()
    }
    
results = {
    "Proposed": evaluate("Decision_Proposed"),
    "Logistic Regression": evaluate("Decision_LR"),
    "Random Forest": evaluate("Decision_RF"),
    "XGBoost": evaluate("Decision_XGB"),
    "Heuristic": evaluate("Decision_Heuristic"),
    "Gradient Boosting": evaluate("Decision_GB"),
    "Isolation Forest": evaluate("Decision_ISO"),
    "Dasu et al.": evaluate("Decision_Dasu_Heuristic"),
    "MLE-RBA(Matiushin et al.)": evaluate("Decision_Matiushin")
}

y_true = df_test["Attack_Flag"]

auc_results = {}

auc_results["Proposed"] = compute_auc_safe(y_true, df_test["Risk_Score"])
auc_results["Logistic Regression"] = compute_auc_safe(y_true, df_test["LR_Prob"])
auc_results["Random Forest"] = compute_auc_safe(y_true, df_test["RF_Prob"])
auc_results["XGBoost"] = compute_auc_safe(y_true, df_test["XGB_Prob"])
auc_results["Heuristic"] = compute_auc_safe(y_true, df_test["Heuristic_Score"])
auc_results["Gradient Boosting"] = compute_auc_safe(y_true, df_test["GB_Prob"])
auc_results["Isolation Forest"] = compute_auc_safe(y_true, df_test["ISO_Score"])
auc_results["Dasu et al."] = compute_auc_safe(y_true, df_test["Dasu_Heuristic_Score"])
auc_results["MLE-RBA(Matiushin et al.)"] = compute_auc_safe(y_true, df_test["Matiushin_Prob"])

plt.figure(figsize=(8,6))

fpr, tpr, auc = safe_roc(y_true, df_test["Risk_Score"])
if fpr is not None and tpr is not None:
    plt.plot(fpr, tpr, label=f"Proposed Framework(AUC={auc:.4f})")
    
    
if "LR_Prob" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["LR_Prob"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle="--" ,label=f"Logistic Regression (AUC={auc:.4f})")

if "RF_Prob" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["RF_Prob"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle="--" ,label=f"Random Forest (AUC={auc:.4f})")

if "XGB_Prob" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["XGB_Prob"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle=":" ,label=f"XGBoost (AUC={auc:.4f})")

if "Heuristic_Score" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["Heuristic_Score"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle="-." ,label=f"Heuristic (AUC={auc:.4f})")
        
if "GB_Prob" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["GB_Prob"])
    # if fpr is not None and tpr is not None:
        # plt.plot(fpr, tpr, linestyle=":" ,label=f"Gradient Boosting (AUC={auc:.4f})")
        
if "ISO_Score" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["ISO_Score"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle="-." ,label=f"Isolation Forest (AUC={auc:.4f})")
        
if "Dasu_Heuristic_Score" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["Dasu_Heuristic_Score"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle=":" ,label=f"Dasu Heuristic (AUC={auc:.4f})")
        
if "Matiushin_Prob" in df_test.columns:
    fpr, tpr, auc = safe_roc(y_true, df_test["Matiushin_Prob"])
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, linestyle="--" ,label=f"MLE-RBA(Matiushin et al.) (AUC={auc:.4f})")
        
plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random Guess (AUC=0.5)")        

scores = df_test["Risk_Score"]
pred_tau2 = scores >= tau2_actual

tp = ((pred_tau2 == True) & (df_test["Attack_Flag"] == 1)).sum()
fp = ((pred_tau2 == True) & (df_test["Attack_Flag"] == 0)).sum()
tn = ((pred_tau2 == False) & (df_test["Attack_Flag"] == 0)).sum()
fn = ((pred_tau2 == False) & (df_test["Attack_Flag"] == 1)).sum()

tpr_tau2 = tp / (tp + fn)
fpr_tau2 = fp / (fp + tn)

print(f"Operating Point at tau2={tau2_actual:.2f}: TPR={tpr_tau2:.4f}, FPR={fpr_tau2:.4f}")
plt.scatter(fpr_tau2, tpr_tau2, color="red", s= 80,label="Policy Blocking Point", zorder=5)
plt.annotate(f"TPR={tpr_tau2:.3f}, FPR={fpr_tau2:.3f}", (fpr_tau2, tpr_tau2), textcoords="offset points", xytext=(15,-10), ha='center', fontsize=9)

pred_tau1 = scores >= tau1_actual

tp1 = ((pred_tau1 == True) & (df_test["Attack_Flag"] == 1)).sum()
fp1 = ((pred_tau1 == True) & (df_test["Attack_Flag"] == 0)).sum()
tn1 = ((pred_tau1 == False) & (df_test["Attack_Flag"] == 0)).sum()
fn1 = ((pred_tau1 == False) & (df_test["Attack_Flag"] == 1)).sum()

tpr_tau1 = tp1 / (tp1 + fn1)
fpr_tau1 = fp1 / (fp1 + tn1)

pi = df_test["Attack_Flag"].mean()
C_FP = 5
C_FN = 10

fpr_vals = np.linspace(0, 1, 100)
tpr_vals = np.linspace(0, 1, 100)

FPR, TPR = np.meshgrid(fpr_vals, tpr_vals)

COST = (C_FP * FPR * (1 - pi)) + (C_FN * (1 - TPR) * pi)

# contours = plt.contour(FPR, TPR, COST, levels=8, linestyles = "dotted", alpha=0.6)

# plt.clabel(contours, inline=True, fontsize=8, fmt="Cost: %.1f")

print(f"Operating Point at tau1={tau1_actual:.2f}: TPR={tpr_tau1:.4f}, FPR={fpr_tau1:.4f}")
plt.scatter(fpr_tau1, tpr_tau1, color="blue", s= 80,label="Policy Step-Up Point", zorder=5)
# plt.annotate("Step-Up Point", (fpr_tau1, tpr_tau1), textcoords="offset points", xytext=(15,-10), ha='center', fontsize=9)
plt.annotate(f"TPR={tpr_tau1:.3f}, FPR={fpr_tau1:.3f}", (fpr_tau1, tpr_tau1), textcoords="offset points", xytext=(15,-10), ha='center', fontsize=9)
plt.fill_betweenx([tpr_tau2, tpr_tau1], fpr_tau2, fpr_tau1, alpha=0.1, label="Policy Region")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison Across Models")
# plt.xlim([0, 0.3])
# plt.ylim([0.7, 1])
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparison_across_models.png")
plt.show()

eer_results = {}

eer_results["Proposed Framework"] = compute_eer(y_true, df_test["Risk_Score"])
eer_results["Logistic Regression"] = compute_eer(y_true, df_test["LR_Prob"])
eer_results["Random Forest"] = compute_eer(y_true, df_test["RF_Prob"])
eer_results["XGBoost"] = compute_eer(y_true, df_test["XGB_Prob"])
eer_results["Heuristic"] = compute_eer(y_true, df_test["Heuristic_Score"])
eer_results["Gradient Boosting"] = compute_eer(y_true, df_test["GB_Prob"])
eer_results["Isolation Forest"] = compute_eer(y_true, df_test["ISO_Score"])
eer_results["Dasu et al."] = compute_eer(y_true, df_test["Dasu_Heuristic_Score"])
eer_results["MLE-RBA(Matiushin et al.)"] = compute_eer(y_true, df_test["Matiushin_Prob"])

print("\nEqual Error Rates (EER) and Thresholds:")
for model, (eer, threshold, _, _) in eer_results.items():
    print(f"{model}: EER={eer:.4f}, Threshold={threshold:.4f}")

plt.figure(figsize=(8,5))
for model, (eer, threshold, fpr, tpr) in eer_results.items():
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, label=f"{model} (EER={eer:.4f})")
        plt.scatter(threshold, 1-eer, label=f"{model} EER Point", s=80)

plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random Guess (EER=0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves with EER Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curves_with_eer_points.png")
plt.show()



plt.figure(figsize=(8,5))

sns.kdeplot(df_test["Risk_Score"], label="Proposed Framework", fill=True,alpha=0.5)
sns.kdeplot(df_test["LR_Cal"], label="Logistic Regression", fill=True,alpha=0.5)
sns.kdeplot(df_test["RF_Cal"], label="Random Forest", fill=True,alpha=0.5)
sns.kdeplot(df_test["XGB_Cal"], label="XGBoost", fill=True,alpha=0.5)
sns.kdeplot(df_test["Heuristic_Cal"], label="Heuristic", fill=True,alpha=0.5)
sns.kdeplot(df_test["GB_Cal"], label="Gradient Boosting", fill=True,alpha=0.5)
sns.kdeplot(df_test["ISO_Cal"], label="Isolation Forest", fill=True,alpha=0.5)
sns.kdeplot(df_test["Dasu_Heuristic_Cal"], label="Dasu Heuristic", fill=True,alpha=0.5)
sns.kdeplot(df_test["Matiushin_Cal"], label="MLE-RBA(Matiushin et al.)", fill=True,alpha=0.5)

plt.axvline(tau1_actual, color="blue", linestyle="--", label="Policy Step-Up Threshold")
plt.axvline(tau2_actual, color="red", linestyle="--", label="Policy Blocking Threshold")

plt.xlabel("Calibrated Risk Score")
plt.ylabel("Density")
plt.title("Density Boundary Distribution Across Models")

plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("density_boundary_distribution_across_models.png")
plt.show()

region_df=pd.DataFrame({
    "Proposed Framework": region_counts(df_test["Risk_Score"]),
    "Logistic Regression": region_counts(df_test["LR_Cal"]),
    "Random Forest": region_counts(df_test["RF_Cal"]),
    "XGBoost": region_counts(df_test["XGB_Cal"]),
    "Heuristic": region_counts(df_test["Heuristic_Cal"]),
    # "Gradient Boosting": region_counts(df_test["GB_Cal"]),
    "Isolation Forest": region_counts(df_test["ISO_Cal"]),
    "Dasu et al.": region_counts(df_test["Dasu_Heuristic_Cal"]),
    "MLE-RBA(Matiushin et al.)": region_counts(df_test["Matiushin_Cal"])
}).T

print(region_df.round(4))

region_df.plot(kind="bar", stacked=True, figsize=(8,5), colormap="Set2")
plt.title("Proportion of Decisions in Each Policy Region")
plt.xlabel("Model")
plt.ylabel("Proportion of Samples")
plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=45)
# plt.legend(title="Policy Region")
# plt.legend(title='Policy Region', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.legend(title='Policy Region', loc='lower center', bbox_to_anchor=(0.5, 1.10), ncol=3)

# plt.ylim(0, 1.3) 
# plt.legend(title='Policy Region', loc='upper right', frameon=True)


plt.tight_layout()
plt.savefig("proportion_of_decisions_in_each_policy_region.png")
plt.show()

print("\n == Prposed Framework Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_Proposed"))

print("\n == Logistic Regression Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_LR"))

print("\n == Random Forest Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_RF"))

print("\n == XGBoost Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_XGB"))

print("\n == Heuristic Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_Heuristic"))

print("\n == Gradient Boosting Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_GB"))

print("\n == Isolation Forest Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_ISO"))

print("\n == Dasu Heuristic Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_Dasu_Heuristic"))

print("\n == MLE-RBA(Matiushin et al.) Confusion Matrix ==")
print(region_confusion_matrix(df_test,"Decision_Matiushin"))

cm = region_confusion_matrix_normalized(df_test,"Decision_Proposed")

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues")

plt.title("Proposed Framework: Region-wise Composition")
plt.ylabel("Policy Region")
plt.xlabel("Class Proportion")

plt.tight_layout()
plt.savefig("proposed_framework_region_wise_composition.png")
plt.show()

results_df = pd.DataFrame(results).T.round(4)
results_df["AUC"] = pd.Series(auc_results).round(4)
results_df["Cost per Detected Attack"] = results_df["Avg Cost"] / (results_df["Recall"])
results_df["Detection Efficiency_1"] = results_df["Recall"]/ results_df["Avg Cost"]
results_df["Detection Efficiency"] = results_df["Block Rate"] / results_df["Avg Cost"]

print(results_df)

results_df.to_csv("final_model_comparison_results.csv")






