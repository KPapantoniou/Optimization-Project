import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('results.csv')
data = data.dropna(axis=1, how='all')

plt.figure(figsize=(10, 6))
plt.boxplot(data.values, notch=True, patch_artist=True, labels=data.columns)

plt.title("Solution Values", fontsize=16)
plt.xlabel("Algorithms", fontsize=14)
plt.ylabel("$f_{best}$", fontsize=14)

plt.savefig("solution_boxplot.png", dpi=300)
plt.show()