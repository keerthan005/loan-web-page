from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    Gender = request.form['Gender']
    Married = request.form['Married']
    Education = request.form['Education']
    Self_Employed = request.form['Self_Employed']
    Property_Area = request.form['Property_Area']
    LoanAmount = float(request.form['LoanAmount'])
    Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
    Credit_History = float(request.form['Credit_History'])

    # Feature engineering
    Is_Graduate = 1 if Education == 'Graduate' else 0
    Is_Self_Employed = 1 if Self_Employed == 'Yes' else 0
    Is_Married = 1 if Married == 'Yes' else 0
    Is_Male = 1 if Gender == 'Male' else 0

    LoanAmount = np.log1p(LoanAmount)
    Loan_Per_Term = LoanAmount / Loan_Amount_Term

    # Create input array
    input_data = np.array([LoanAmount, Loan_Amount_Term, Credit_History,
                           Is_Graduate, Is_Self_Employed, Is_Married, Is_Male,
                           Loan_Per_Term,
                           1 if Property_Area == 'Semiurban' else 0,
                           1 if Property_Area == 'Urban' else 0])

    # Scale numeric features
    input_scaled = scaler.transform([input_data[:3] + [input_data[-1]]])
    input_data[:3] = input_scaled[0][:3]
    input_data[-1] = input_scaled[0][3]

    # Final prediction
    prediction = model.predict([input_data])[0]
    result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
