import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

df_test["LR_Prob"] = lr.predict_proba(X_test)[:,1]
df_test["RF_Prob"] = rf.predict_proba(X_test)[:,1]

df_test["LR_Cal"] = calibrate(df_test["LR_Prob"])
df_test["RF_Cal"] = calibrate(df_test["RF_Prob"])

df_test["Decision_Proposed"]=df_test["Risk_Score"].apply(decision)
df_test["Decision_LR"] = df_test["LR_Cal"].apply(decision)
df_test["Decision_RF"] = df_test["RF_Cal"].apply(decision)

def evaluate(decision_col):
    benign = df_test[df_test["Attack_Flag"]==0]
    attack = df_test[df_test["Attack_Flag"]==1]
    
    print("Decision col", decision_col)
        
    decisions = df_test[decision_col]    
    
        
    return{
        "Avg Cost": decisions.map(cost_map).mean(),
        "Step-Up Rate": (decisions == "STEP_UP").mean(),
        "Block Rate": (decisions == "BLOCK").mean(),
        "False Block Rate": (benign[decision_col]=="BLOCK").mean(),
        "Benign Step-Up Rate": (benign[decision_col]=="STEP_UP").mean(),
        "Attack Detection Rate": (attack[decision_col]=="BLOCK").mean()
    }
    
results = pd.DataFrame({
    "Proposed Framework": evaluate("Decision_Proposed"),
    "Logistic Regression": evaluate("Decision_LR"),
    "Random Forest": evaluate("Decision_RF")
}).T
print(results.round(4))

results[[
    "Attack Detection Rate","Step-Up Rate","False Block Rate","Avg Cost"
]].plot(kind="bar", figsize=(10,6))

plt.title("Friction Vs Security Comparison")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("friction_vs_security_comparison.png")
plt.show()

plt.figure(figsize=(7,5))
costs, recalls = [], []

for t in np.linspace(0.01, 0.4, 50):
    
    decisions = df_test["Risk_Score"].apply(lambda r: "BLOCK" if r>t else "ALLOW")
    cost = decisions.map(cost_map).mean()
    recall = (df_test[df_test["Attack_Flag"]==1]["Risk_Score"] >= t).mean()
    
    costs.append(cost)
    recalls.append(recall)
    
plt.scatter(costs, recalls, alpha=0.4, label="Policy Sweep")
plt.scatter(results.loc["Proposed Framework","Avg Cost"], results.loc["Proposed Framework","Attack Detection Rate"], marker="x", s=100, label="Proposed Framework")
plt.scatter(results.loc["Logistic Regression","Avg Cost"], results.loc["Logistic Regression","Attack Detection Rate"], marker="o", s=100, label="Logistic Regression")
plt.scatter(results.loc["Random Forest","Avg Cost"], results.loc["Random Forest","Attack Detection Rate"], marker="s", s=100, label="Random Forest")
plt.xlabel("Average Authentication Cost")
plt.ylabel("Attack Detection Rate")
plt.title("Cost vs. Security Tradeoff")
plt.legend()
plt.grid(True)
plt.savefig("cost_vs_security_comparison.png", dpi=300)
plt.show()

decision_dist = pd.DataFrame({"Proposed Framework": df_test["Decision_Proposed"].value_counts(normalize=True),
                              "Logistic Regression": df_test["Decision_LR"].value_counts(normalize=True),
                              "Random Forest": df_test["Decision_RF"].value_counts(normalize=True)}).fillna(0)

decision_dist.T.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("Distribution of Authentication Decisions")
plt.xticks(rotation=0)
plt.ylabel("Proportion of Sessions")
plt.legend(title="Decision")
plt.tight_layout()
plt.savefig("decision_distribution_comparison.png", dpi=300)
plt.show()


