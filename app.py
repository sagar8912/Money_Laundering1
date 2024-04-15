from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
with open('money_laundering.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    tempo = float(request.form['tempo'])
    amount = float(request.form['amount'])
    initial_balance_c1 = float(request.form['initial_balance_c1'])
    new_balance_c1 = float(request.form['new_balance_c1'])
    initial_balance_c2 = float(request.form['initial_balance_c2'])
    new_balance_c2 = float(request.form['new_balance_c2'])
    type_CASH_IN = int(request.form.get('type_CASH_IN', 0))
    type_CASH_OUT = int(request.form.get('type_CASH_OUT', 0))
    type_DEBIT = int(request.form.get('type_DEBIT', 0))
    type_PAYMENT = int(request.form.get('type_PAYMENT', 0))
    type_TRANSFER = int(request.form.get('type_TRANSFER', 0))

    # Convert input values to DataFrame
    data = pd.DataFrame({
        'tempo': [tempo],
        'amount': [amount],
        'initial_balance_c1': [initial_balance_c1],
        'new_balance_c1': [new_balance_c1],
        'initial_balance_c2': [initial_balance_c2],
        'new_balance_c2': [new_balance_c2],
        'type_CASH_IN': [type_CASH_IN],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_DEBIT': [type_DEBIT],
        'type_PAYMENT': [type_PAYMENT],
        'type_TRANSFER': [type_TRANSFER]
    })

    # Make prediction
    prediction = model.predict(data)[0]

    # Convert prediction to "YES" or "NO"
    prediction_text = "YES" if prediction == 1 else "NO"

    # Pass prediction to result page
    return render_template('result.html', prediction_text=prediction_text, prediction_class="yes" if prediction == 1 else "no")


if __name__ == '__main__':
    app.run(debug=True, port=1111)
