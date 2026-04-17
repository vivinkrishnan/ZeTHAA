import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

df=pd.read_csv("synthetic_authentication_dataset.csv")

df["Attack_Flag"]=(df["Label"]=="Attack").astype(int)

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

features = [
"Geo_Anomaly",
"Travel_Anomaly",
"Timezone_Anomaly",
"Temporal_Anomaly",
"Fingerprint_Mismatch",
"Token_Context_Mismatch",
"Repeated_Attacker_IP"
]

# X=df[features].astype(int)
# y=df["Attack_Flag"]

alpha_prior = 1
beta_prior = 1

# weights=mutual_info_classif(X,y)
weights = []
base_attack_rate = df["Attack_Flag"].mean()

for f in features:
    # p_attack_given_signal = df[df[f] == 1]["Attack_Flag"].mean()
    
    signal_rows = df[df[f] == 1]
    attacks = signal_rows["Attack_Flag"].sum()
    benign = len(signal_rows) - attacks
    
    posterior = (alpha_prior + attacks) / (alpha_prior + beta_prior + attacks + benign)
        
    weight = max(posterior - base_attack_rate, 0)
    weights.append(weight)

print(weights)

weights = np.array(weights)

if weights.sum() > 0:
    weights = weights / weights.sum()

# weights = weights / weights.sum()

weights_df = pd.DataFrame({
"Feature":features,
"Weight":weights
})

weights_df.to_csv("calibrated_weights.csv",index=False)

print("Weights computed", weights_df)