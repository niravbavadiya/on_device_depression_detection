import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_pruned\INS-W-sample_1\FeatureData\norm\location.csv")
sns.heatmap(df.isna(), cbar=False); plt.title("missingness after prune")
plt.show()

print(df.var().describe())              # confirm no near-zeros
sns.clustermap(df.corr(), figsize=(10,10), vmax=.9, vmin=-.9)