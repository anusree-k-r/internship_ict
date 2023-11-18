from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)

file1 = open('model.pkl', 'rb')
rf = pickle.load(file1)

@app.route('/', methods=['GET', 'POST'])

def predict():
    prediction=''
    if request.method == 'POST':
        my_dict = request.form
        Num_Bank_Accounts = float(my_dict['Num_Bank_Accounts'])
        Num_Credit_Card = float(my_dict['Num_Credit_Card'])
        Interest_Rate = float(my_dict['Interest_Rate'])
        Num_of_Loan = float(my_dict['Num_of_Loan'])
        Delay_from_due_date = float(my_dict['Delay_from_due_date'])
        Num_of_Delayed_Payment = float(my_dict['Num_of_Delayed_Payment'])
        Num_Credit_Inquiries = float(my_dict['Num_Credit_Inquiries'])
        Credit_Mix = int(my_dict['Credit_Mix'])
        Outstanding_Debt = float(my_dict['Outstanding_Debt'])
        Credit_History_Age = float(my_dict['Credit_History_Age'])
        
        
        # Create DataFrame from input features
        input_data = pd.DataFrame({
            'Num_Bank_Accounts': [Num_Bank_Accounts],
            'Num_Credit_Card': [Num_Credit_Card],
            'Interest_Rate': [Interest_Rate],
            'Num_of_Loan': [Num_of_Loan],
            'Delay_from_due_date': [Delay_from_due_date],
            'Num_of_Delayed_Payment': [Num_of_Delayed_Payment],
            'Num_Credit_Inquiries': [Num_Credit_Inquiries],
            'Credit_Mix': [Credit_Mix],
            'Outstanding_Debt': [Outstanding_Debt],
            'Credit_History_Age': [Credit_History_Age]
        })
        
        prediction = rf.predict(input_data)[0]
        prediction=int(prediction)
        if(prediction==2):
            prediction='Good Credit Score'
        if(prediction==1):
            prediction='Standard Credit Score'
        if(prediction==0):
            prediction='Poor Credit Score'
        print(prediction)
    
        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)
