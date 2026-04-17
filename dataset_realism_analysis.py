import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("synthetic_authentication_dataset.csv")

df["Time_of_Access"] = pd.to_datetime(df["Time_of_Access"])

# ---------------------------------------------------
# Login Hour Distribution
# ---------------------------------------------------

df["Login_Hour"] = df["Time_of_Access"].dt.hour

plt.figure()
sns.histplot(df["Login_Hour"], bins=24)

plt.title("Login Hour Distribution")
plt.xlabel("Hour of Day")

plt.savefig("realism_login_hours.png")


# ---------------------------------------------------
# Weekday vs Weekend Activity
# ---------------------------------------------------

weekday_counts = df.groupby(["Weekday","Label"]).size().unstack()

weekday_counts.plot(kind="bar")

plt.title("Login Activity by Weekday")

plt.savefig("realism_weekday_activity.png")


# ---------------------------------------------------
# Travel Distribution
# ---------------------------------------------------

travel = df.groupby("Travel_Status").size()

travel.plot(kind="bar")

plt.title("Travel vs Home Activity")

plt.savefig("realism_travel_distribution.png")


# ---------------------------------------------------
# Device Lifecycle Events
# ---------------------------------------------------

device_counts = df.groupby("Device_Model")["User_ID"].nunique()

device_counts.plot(kind="bar")

plt.title("Device Usage Distribution")

plt.savefig("realism_device_usage.png")


# ---------------------------------------------------
# OS Version Distribution
# ---------------------------------------------------

os_dist = df["OS_Version"].value_counts()

os_dist.plot(kind="bar")

plt.title("OS Version Distribution")

plt.savefig("realism_os_versions.png")


# ---------------------------------------------------
# Application Version Rollout
# ---------------------------------------------------

app_dist = df.groupby("Application_Version").size()

app_dist.plot(kind="bar")

plt.title("Application Version Distribution")

plt.savefig("realism_app_versions.png")


# ---------------------------------------------------
# Risk Score Distribution
# ---------------------------------------------------

plt.figure()

sns.histplot(
    df[df["Label"]=="Benign"]["Risk_Score"],
    color="blue",
    label="Benign",
    alpha=0.5
)

sns.histplot(
    df[df["Label"]=="Attack"]["Risk_Score"],
    color="red",
    label="Attack",
    alpha=0.5
)

plt.legend()
plt.title("Risk Score Separation")

plt.savefig("realism_risk_distribution.png")


# ---------------------------------------------------
# Attack Type Distribution
# ---------------------------------------------------

attack_dist = df["Attack_Type"].value_counts()

attack_dist.plot(kind="bar")

plt.title("Attack Type Distribution")

plt.savefig("realism_attack_distribution.png")

print("Dataset realism analysis complete.")