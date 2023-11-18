import pickle
import pandas as pd
import numpy as np

data=pd.read_csv('credit.csv')

data.drop(columns=['ID','Name','SSN','Type_of_Loan'], inplace=True)
data.drop(columns=['Customer_ID','Month', 'Age', 'Occupation', 'Annual_Income','Monthly_Inhand_Salary','Changed_Credit_Limit','Credit_Utilization_Ratio',
       'Payment_of_Min_Amount', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance' ], inplace=True)

credit_mix = {
    'Standard': 1,
    'Good': 2,
    'Bad': 0
}
data.Credit_Mix.replace(credit_mix, inplace=True)
data.Credit_Mix = data.Credit_Mix.astype(int)

credit_score = {
    'Standard': 1,
    'Poor': 0,
    'Good': 2
}
data.Credit_Score.replace(credit_score, inplace=True)
data.Credit_Score = data.Credit_Score.astype(int)

X5=data.drop(['Credit_Score'],axis=1)
Y5=data['Credit_Score']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,Y_train,Y_test=train_test_split(X5,Y5,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model_rfc=model.fit(X_train, Y_train)
y_pred_rd_clf = model_rfc.predict(X_test)


acc_rd_clf = accuracy_score(Y_test, y_pred_rd_clf)
print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")

# Saving model to disk
pickle.dump(model,open('model.pkl','wb'))