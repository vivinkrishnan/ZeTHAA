import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

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
    
    rows.append({
        "Attribute": attr, 
        "Distinct_Values": df[attr].nunique(), 
        "Information_Gain": round(ig, 4), 
        "Penalty": round(penalty, 4), 
        "P(Attack| Attribute)": round(p_attack, 4)          
    })
    
results = pd.DataFrame(rows)
results["Weight"] = results["Information_Gain"] / results["Information_Gain"].sum()
results.sort_values("Weight", ascending=False, inplace=True)

results.to_csv("attribute_analysis_table.csv", index=False)

print(results)

plt.figure(figsize=(7,5))

plt.barh(results["Attribute"], results["Weight"], color="skyblue")
plt.xlabel("Attribute Weight")
plt.title("Attribute Importance Analysis")

plt.gca().invert_yaxis()

plt.savefig("attribute_weights.png",dpi=300)