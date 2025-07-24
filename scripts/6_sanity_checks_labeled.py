import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, glob, pathlib

files = glob.glob(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\4_data_labeled\INS-W-sample_1\FeatureData\norm\location.csv")
df    = pd.concat([pd.read_csv(f, usecols=["pid","dep_weekly",
                                           "na_ema","dep_static"])
                   for f in files])

print(df.dep_weekly.value_counts(dropna=False))
sns.histplot(df.na_ema.dropna()); plt.title("Negative affect"); plt.show()