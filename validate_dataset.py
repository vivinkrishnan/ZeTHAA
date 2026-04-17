import pandas as pd

df=pd.read_csv("synthetic_authentication_dataset.csv")

assert df["Trust_Score"].between(0,1).all()
assert df["Application_Authenticity_Data"].notnull().all()
assert df["Weekday"].between(0,6).all()
assert df["Is_Weekend"].isin([True,False]).all()
assert df["OS_Version"].min() >= 10
assert df["Timezone_Shift"].between(-12,12).all()

print("Dataset validation passed")