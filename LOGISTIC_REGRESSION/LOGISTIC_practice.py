import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
 

df = pd.DataFrame({
    "StudyHours":[2,3,4,5,6,7,8,1,9,4,6,3,8,5,2],
    "Attendance":[55,60,65,70,75,80,90,50,92,68,78,58,88,72,52],
    "Assignments":[2,3,4,5,6,7,8,1,9,5,7,3,8,6,2],
    "SleepHours":[5,6,6,7,7,7,8,5,8,6,7,5,8,7,5],
    "Pass":[0,0,0,1,1,1,1,0,1,0,1,0,1,1,0]
})


X = df.drop("Pass",axis=1)
y = df["Pass"]
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.3)

model = LogisticRegression()

model.fit(X_train,y_train)

prid = model.predict(X_test)

print("MODEL PRIDICTION RESULT : ",prid)

print("\n ACTUAL VALUES : ",y_test)

ACC = accuracy_score(y_test,prid)

print("ACCURACY OF THIS MODEL  ",ACC)


cm = confusion_matrix(y_test,prid)

print("confusion_matrix : ",cm)

clas = classification_report(y_test,prid)

model = LogisticRegression()
model.fit(X,y)
# stu_data = [[2,60,5,8]]

stu_data = pd.DataFrame({
    "StudyHours":[1],
    "Attendance":[55],
    "Assignments":[1],
    "SleepHours":[5]
})

model.predict(stu_data)
model.predict_proba(stu_data)

stu_pass_fail = model.predict(stu_data)
prob = model.predict_proba(stu_data)

print("probability of student to getting failor fass : ",prob)
print("STUDENT PASS/FAIL : ",stu_pass_fail)
