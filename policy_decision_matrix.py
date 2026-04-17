import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
"ALLOW":[59872,6229],
"STEP_UP":[13973,3512],
"BLOCK":[5985,26521]
}

df = pd.DataFrame(data, index=["Benign","Attack"])
print(df)
df_rate = df.div(df.sum(axis=1), axis=0)

print(df_rate)

plt.figure(figsize=(6,4))

sns.heatmap(
    df_rate,
    annot=True,
    cmap="YlOrRd", 
    fmt=".2%", 
    cbar=True
)

plt.title("Policy Decision Matrix")
plt.xlabel("Decision")
plt.ylabel("Event Type")

plt.savefig("policy_decision_matrix.png")
plt.show()
            
            