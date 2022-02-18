import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("Swaps.csv")
arr = df.to_numpy()

pca = PCA(n_components = len(arr[0]) - 1)
pca.fit(arr)
sv = pca.singular_values_
print("Largest Singular Values: ", sv[0], sv[1], sv[2])
print("Largest Relative Power: ", (sv[0] + sv[1] + sv[2]) / sum(sv))

plt.plot(sv)
plt.title("Singular Values")
plt.show()
