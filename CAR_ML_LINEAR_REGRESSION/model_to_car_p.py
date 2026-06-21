import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\MachineLearning\\CAR_ML_LINEAR_REGRESSION\\car_dataset.csv")

# print(df.columns)

# plt.scatter(df["PRICE"],df["MANUFACTURE"])
# plt.show()

df_encoded = pd.get_dummies(df,columns=["FUEL_TYPE","ACCIDENT_HISTORY"])

X = df_encoded.drop(columns=["CAR_COMPANY","TYRE_COMPANY","BODY","PRICE"])
y = df_encoded["PRICE"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

model = LinearRegression()
model.fit(X_train,y_train)

y_pridict = model.predict(X_test)

print(y_pridict)

rsme =root_mean_squared_error(y_pridict,y_test)
print("root_mean_squared_error: ",rsme)





# Step 1: enter your raw values here
new_data = {
    "MANUFACTURE": 2020,
    "PRE_OWNERS": 1,
    "SEATER": 5,
    "MILEAGE": 18,
    "FUEL_TYPE": "PETROL",        # one of: PETROL, DIESEL, CNG, ELECTRIC
    "ACCIDENT_HISTORY": "NO"      # one of: YES, NO
}

# Step 2: convert to a DataFrame
new_df = pd.DataFrame([new_data])

# Step 3: apply the same one-hot encoding as training data
new_df_encoded = pd.get_dummies(new_df, columns=["FUEL_TYPE", "ACCIDENT_HISTORY"])

# Step 4: align columns with X (adds any missing dummy columns as 0, drops extras, keeps order same)
new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)

# Step 5: predict
predicted_price = model.predict(new_df_encoded)

print("Predicted Price:", predicted_price[0])