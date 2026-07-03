from sklearn.cluster import KMeans
import pandas as pd

df = pd.DataFrame({
    "Income":[20,60,24,25,80,82,85,21]
})

kmeans = KMeans(n_clusters=2, random_state=42)

kmeans.fit(df)
print(kmeans.cluster_centers_)
print(kmeans.labels_)