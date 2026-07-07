# import pandas as pd 
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# df = pd.DataFrame({
#     "Age": [18,20,22,45,47,50,22,25,50,18],
#     "Gender": ["Male","Male","Male","Female","Female","Female","Male","Female","Male","Male"],
#     "Score": [90,85,50,30,35,25,28,35,23,20]
# })

# df_encoded = pd.get_dummies(df,columns=["Gender"])

# # print(df_encoded)
# scaler = StandardScaler()

# df_scaled = scaler.fit_transform(df_encoded)

# kmeans = KMeans(n_clusters=2,random_state=42)

# kmeans.fit(df_scaled)


# print(kmeans.cluster_centers_)

# print(kmeans.labels_)
# print(kmeans.predict([[30,0,1,80]]))

# df["group"] = kmeans.labels_

# print(df)

# print(df.groupby("group").mean(numeric_only=True))
# print(df.groupby("group").describe())


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({
    "Income":[20,22,24,25,80,82,55,25,46,79,29,88,61,49,6,12,59,76,2,59,65,11,99,82,30,110,40,62,16,97,35,88,36,79,20,56,7,89,42]
})

print(df["Income"].count())

wcss = []

for i in range(1,df["Income"].count()):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(df)

    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,df["Income"].count()), wcss, marker="o")

plt.xlabel("K")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()