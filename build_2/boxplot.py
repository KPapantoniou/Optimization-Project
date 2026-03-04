import matplotlib.pyplot as plt
import pandas as pd

files = {
    "NewtonTR": "output_NewtonTR_train.txt",
    "BFGSWolfe": "output_BFGSWolfe_train.txt",
    "NelderMead": "output_NelderMead_train.txt",
    "GA": "output_GA_train.txt",
    "PSO": "output_PSO_train.txt",
}

data = {}
for name, path in files.items():
    df = pd.read_csv(path, sep=r"\s+", header=None)
    data[name] = df.iloc[:, 1].values

plt.figure(figsize=(10, 6))
plt.boxplot(list(data.values()), notch=True, patch_artist=True, labels=list(data.keys()))

plt.title("Solution Values", fontsize=16)
plt.xlabel("Algorithms", fontsize=14)
plt.ylabel("$f_{best}$", fontsize=14)

plt.savefig("solution_boxplot.png", dpi=300)
plt.show()
