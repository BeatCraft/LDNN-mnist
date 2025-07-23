import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

value_map = {
    0: -1.0,
    1: -0.5,
    2: -0.25,
    3: -0.125,
    4: 0.0,
    5: 0.125,
    6: 0.25,
    7: 0.5,
    8: 1.0
}

df = pd.read_csv("./wi-fc.csv.4", encoding="utf-8", engine="python", on_bad_lines='skip')
df.head()

df_numeric = df.apply(pd.to_numeric, errors='coerce')
df_mapped = df_numeric.replace(value_map)

plt.figure(figsize=(16, 8))
sns.heatmap(df_mapped, cmap="bwr", center=0, cbar_kws={'label': 'Mapped Value'})
plt.title("Red=Positive, Blue=Negative)")
plt.xlabel("Column Index")
plt.ylabel("Row Index")

heatmap_path = "./hmap4.png"
plt.tight_layout()
plt.savefig(heatmap_path)
plt.close()

heatmap_path


