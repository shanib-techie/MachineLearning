"""1. Train–Test Split kyun zaroori hai?

Tu apni CSV file ko 2 parts me divide karta hai:

Train data → model yahan se seekhta hai
Test data → model ko check karne ke liye (jo usne kabhi nahi dekha)
Problem agar split nahi kiya:

Agar tu same data pe train + test karega → model ratta maar lega (overfitting)

👉 Matlab:

Train pe error low
Real world me fail
Real life example:

Soch tu exam ke exact same questions rat leta hai
→ exam me full marks
→ but real knowledge = zero

Train-test split = practice vs real exam"""

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Step 1: CSV file load karo
# df = pd.read_csv("c:\\Users\\Admin\\OneDrive\\Desktop\\MachineLearning\\ML_practise\\student_score.csv")   # <-- apni file ka naam yahan daal

# # Step 2: Columns check kar (important)
# print(df.head())

# # Step 3: X aur y define karo
# # Yahan column names apne data ke hisaab se change kar
# X = df[["Hours"]]    # input column
# y = df["Score"]      # output column

# # Step 4: Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2
# )

# # Step 5: Model train karo
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Step 6: Prediction
# y_pred = model.predict(X_test)

# # Step 7: MSE calculate
# mse = mean_squared_error(y_test, y_pred)

# # Step 8: Output
# print("Test Index:", y_test.index.tolist())
# print("Actual Values:", list(y_test))
# print("Predicted Values:", y_pred)
# # print("Actual Values:", list(y_test))
# # print("Predicted Values:", y_pred)
# print("MSE:", mse)

# # Step 9: Equation bhi dekh le
# print("Slope (m):", model.coef_[0])
# print("Intercept (c):", model.intercept)




import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.DataFrame({
    "months" : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "sales" : [100,200,280,360,400,450,490,520,540,555,580,600,610,612,620]
})

X = df[["months"]]
y = df["sales"]

X_train , X_Test, y_train , y_test = train_test_split (
      X,y, test_size=0.4
)

model = LinearRegression()
model.fit(X_train,y_train)

y_pridict = model.predict(X_Test)

mse = mean_absolute_error(y_test,y_pridict)


print("Test Index:", y_test.index.tolist())

print("ACTUAL VALUE :",list(y_test))
print("PRIDICT VALUE : ",list(y_pridict))
print("MSE : ",mse)

