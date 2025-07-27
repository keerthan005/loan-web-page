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
    try:
        # Get form inputs
        Gender = request.form['Gender']
        Married = request.form['Married']
        Education = request.form['Education']
        Self_Employed = request.form['Self_Employed']
        Property_Area = request.form['Property_Area']
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])

        # Encode categorical variables
        Is_Graduate = 1 if Education == 'Graduate' else 0
        Is_Self_Employed = 1 if Self_Employed == 'Yes' else 0
        Is_Married = 1 if Married == 'Yes' else 0
        Is_Male = 1 if Gender == 'Male' else 0
        Prop_Semiurban = 1 if Property_Area == 'Semiurban' else 0
        Prop_Urban = 1 if Property_Area == 'Urban' else 0

        # Feature transformation
        LoanAmount_log = np.log1p(LoanAmount)
        Loan_Per_Term = LoanAmount_log / Loan_Amount_Term

        # Combine features
        input_data = np.array([LoanAmount_log, Loan_Amount_Term, Credit_History, Loan_Per_Term])
        scaled_features = scaler.transform([input_data])[0]

        # Final feature set
        final_input = np.concatenate((
            scaled_features,
            [Is_Graduate, Is_Self_Employed, Is_Married, Is_Male, Prop_Semiurban, Prop_Urban]
        ))

        # Make prediction
        prediction = model.predict([final_input])[0]
        result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'

        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
