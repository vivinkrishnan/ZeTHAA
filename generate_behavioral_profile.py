import pandas as pd
import numpy as np
from collections import defaultdict

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("synthetic_authentication_dataset.csv")

df["Attack_Flag"] = (df["Label"]=="Attack").astype(int)

# Sort by time for sequential learning
if "Timestamp" in df.columns:
    df = df.sort_values("Timestamp")

# ----------------------------
# PARAMETERS
# ----------------------------
sigma0 = 0.05
gamma = 0.2

tau_learn = 0.6
epsilon = 5

penalty_decay = 0.98
penalty_max = 10

window_size = 5



# Behavioral attributes (signals)
attributes = [
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

# ----------------------------
# PROFILE STORAGE
# ----------------------------
# P_u[user][attribute] = {mu, sigma2, penalty}
P_u = defaultdict(dict)

# ----------------------------
# INITIALIZATION FUNCTION
# ----------------------------
def initialize_profile(user, row):
    for attr in attributes:
        value = row[attr]
        P_u[user][attr] = {
            "mu": value,
            "sigma2": sigma0,
            "penalty": 0.0
        }

def trust_gated_update(user, row, trust_score):

    for attr in attributes:

        a_t = row[attr]

        mu = P_u[user][attr]["mu"]
        sigma2 = P_u[user][attr]["sigma2"]
        penalty = P_u[user][attr]["penalty"]

        # Gating
        if trust_score < tau_learn or penalty > epsilon:
            continue

        # Mean update
        mu_new = (1 - gamma) * mu + gamma * a_t

        # Variance update
        sigma2_new = (1 - gamma) * sigma2 + gamma * (a_t - mu)**2

        P_u[user][attr]["mu"] = mu_new
        P_u[user][attr]["sigma2"] = sigma2_new

# ----------------------------
# WINDOWED UPDATE (Algorithm 3)
# ----------------------------
def windowed_update(user, window_df):

    trust_window = window_df["Trust_Score"].mean()

    for attr in attributes:

        penalty_window = window_df[attr].sum()

        if trust_window < tau_learn or penalty_window > 0:
            continue

        mu = P_u[user][attr]["mu"]

        a_bar = window_df[attr].mean()

        mu_new = (1 - gamma) * mu + gamma * a_bar

        P_u[user][attr]["mu"] = mu_new

# ----------------------------
# PENALTY UPDATE + DECAY
# ----------------------------
def update_penalty(user, row):

    for attr in attributes:

        if row[attr] == 1:
            P_u[user][attr]["penalty"] += 1

        # Decay
        P_u[user][attr]["penalty"] *= penalty_decay

        # Clip
        P_u[user][attr]["penalty"] = min(P_u[user][attr]["penalty"], penalty_max)

# ----------------------------
# NORMALIZED PENALTY
# ----------------------------
def get_normalized_penalty(user, attr):
    return P_u[user][attr]["penalty"] / penalty_max

def compute_risk(user):

    risk = 0

    for attr in attributes:

        mu = P_u[user][attr]["mu"]
        penalty = get_normalized_penalty(user, attr)

        # Combine behavior + penalty
        risk += mu + 0.5 * penalty

    # Normalize to [0,1]
    risk = risk / len(attributes)

    return min(risk, 1.0)

# ----------------------------
# MAIN LOOP
# ----------------------------

def run_behavioral_engine():

    user_buffers = defaultdict(list)
    risk_scores = []

    for _, row in df.iterrows():
        user = row["User_ID"]

        if user not in P_u:
            initialize_profile(user, row)        
        
        update_penalty(user,row)
        
        trust_score = row.get("Trust_Score", 0.7)
        
        trust_gated_update(user, row, trust_score)
        
        user_buffers[user].append(row)
        
        if(len(user_buffers[user])>=window_size):
            window_df = pd.DataFrame(user_buffers[user][-window_size:])
            windowed_update(user, window_df)
        
        risk = compute_risk(user)
        risk_scores.append(risk)
        
    df["Behavioral_Risk"] = risk_scores

    profiles = []

    for user, attrs in P_u.items():
        for attr, vals in attrs.items():
            profiles.append({
                "User_ID": user,
                "Attribute": attr,
                "Mu": vals["mu"],
                "Sigma2": vals["sigma2"],
                "Penalty": vals["penalty"]
            })

    df_profiles = pd.DataFrame(profiles)

    print("\n Behavioral Profiles:\n")
    print(df_profiles.head())
    
run_behavioral_engine()