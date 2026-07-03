import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\MachineLearning\\LOGISTIC_REGRESSION\\MALL_CUSTOMER_BUY_OR_NOT_PRIDICTION\\mall_data.csv")

# print(df.columns)



# df["buy"] = (
#     (df["Age"] >= 50)
#     &
#     (
#         (df["Annual Income (k$)"] >= 55)
#         |
#         (df["Spending Score (1-100)"] >= 55)
#     )
# ).astype(int)

# df["buy"] = (
#     (df["Age"] >= 50)
#        &
#     (   
#     (df["Annual Income (K$)"] >= 55)
#        |
#     (df["Spending Score (1-100)"] >=55)      
#     )
# )\\

df_encoded = pd.get_dummies(df,columns=["Gender"])
X = df_encoded.drop(columns=["Unnamed: 0","CustomerID","buy"])
y= df_encoded["buy"]
print(df_encoded)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model = LogisticRegression()
model.fit(X_train,y_train)
prid = model.predict(X_test)

print("ACTUAL VALUE OF BUT :/ ",y_test)
# 
print("PRIDICT VALUE OF Y : ",prid)

acc = accuracy_score(y_test,prid)
print("\nACCURACY OF MODEL : ",acc)

cm = confusion_matrix(y_test,prid)

print("\nCONFUSION METRIC ",cm)


# print(X.columns)

# new_customer = pd.DataFrame(
#     [[50, 60, 88, 0,1]],
#     columns=X.columns
# )

new_customer = pd.DataFrame  ({
      "Age" : [29],
      "Annual Income (k$)" : [500],
      "Spending Score (1-100)" : [90],
     "Gender_Female" : [0],
      "Gender_Male" : [1]
})
model = LogisticRegression()
model.fit(X,y)
buy_r_not = model.predict(new_customer)

print("NEW CUSTOMER WITH THIS DETAIK LIKELY TO ",buy_r_not)
p_of_b_n = model.predict_proba(new_customer)
print("PROBABILY OF NW CUSTOMER FOR BUY OR NOT IS ", p_of_b_n)