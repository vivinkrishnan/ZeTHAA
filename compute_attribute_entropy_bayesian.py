import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

df=pd.read_csv("synthetic_authentication_dataset.csv")
df["Attack_Flag"]=(df["Label"]=="Attack").astype(int)

attributes=[
    "Device_Model",
    "Geo_Location",
    "Travel_Status",
    "Timezone_Shift",
    "Platform",
    "OS_Version",
    "IP_Address",
    "Application_Version",
    "Application_Authenticity_Data",
    "User_Agent",
    "Screen_Resolution",
    "Request_Type"
]

rows = []

for attr in attributes:
    ig = mutual_info_score(df[attr], df["Attack_Flag"])

    attack_probs = df.groupby(attr)["Attack_Flag"].mean()

    penalty = attack_probs.max()
    p_attack = attack_probs.mean()
    
    alpha = 1
    beta = 1
    
    successes = df.groupby(attr)["Attack_Flag"].sum()
    failures = df.groupby(attr)["Attack_Flag"].count() - successes
    
    posterior_means = (alpha + successes) / (alpha + beta + successes + failures)
    
    bayesian_weight = posterior_means.mean()
    
    rows.append({
        "Attribute": attr, 
        "Distinct_Values": df[attr].nunique(), 
        "Information_Gain": ig, 
        "Penalty": penalty, 
        "P(Attack| Attribute)": p_attack, 
        "Bayesian_Weight": bayesian_weight          
    })
    
results = pd.DataFrame(rows)

results["Entropy_Weight"] = results["Information_Gain"] / results["Information_Gain"].sum()
results.sort_values("Entropy_Weight", ascending=False, inplace=True)

results.to_csv("attribute_entropy_bayesian_analysis.csv", index=False)

print(results)