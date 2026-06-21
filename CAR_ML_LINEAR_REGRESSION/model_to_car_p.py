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





# # Step 1: enter your raw values here
# new_data = {
#     "MANUFACTURE": 2020,
#     "PRE_OWNERS": 1,
#     "SEATER": 5,
#     "MILEAGE": 18,
#     "FUEL_TYPE": "PETROL",        # one of: PETROL, DIESEL, CNG, ELECTRIC
#     "ACCIDENT_HISTORY": "NO"      # one of: YES, NO
# }

def predict_price(manufacture, owners, seater,
                  mileage, fuel, accident):

    car = pd.DataFrame([{
        "MANUFACTURE": manufacture,
        "PRE_OWNERS": owners,
        "SEATER": seater,
        "MILEAGE": mileage,
        "FUEL_TYPE": fuel,
        "ACCIDENT_HISTORY": accident
    }])

    car = pd.get_dummies(
        car,
        columns=["FUEL_TYPE", "ACCIDENT_HISTORY"]
    )

    car = car.reindex(columns=X.columns, fill_value=0)

    return model.predict(car)[0]



manufacture_year = str(input("ENTER THE MANUFACTURE YEAR : "))
owners = int(input("ENTER THE PREVIOUS OWNERS : "))
seater = int(input("ENTER THE SEATER CAPACITY : "))
mileage = int(input("ENTER THE MILEAGE OF VICHLE : "))
FUEL =   str(input("ENTER THE FUEL TYPE  (PETROL/DIESEL/CNG/ELECTRIC) : "))
ACC_HISTORY = str(input("DID ANY PREVIOUS ACCIDENT HISTORY (YES/NO) : "))
price = predict_price(

    manufacture_year, owners, seater, mileage, FUEL, ACC_HISTORY
)

print("Predicted Price:", price)