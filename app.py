from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["Age"])
        income = float(request.form["Income"])
        credit_score = float(request.form["CreditScore"])
        loan_amount = float(request.form["LoanAmount"])
        experience = float(request.form["Experience"])
        education = request.form["Education"]
        employment = request.form["Employment"]

        if income <= 0 or loan_amount <= 0:
            return render_template(
                "index.html",
                prediction="Income and loan amount must be greater than zero."
            )

        loan_income_ratio = loan_amount / income

        input_data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "CreditScore": credit_score,
            "LoanAmount": loan_amount,
            "Experience": experience,
            "LoanIncomeRatio": loan_income_ratio,
            "Education": education,
            "Employment": employment
        }])

        result = model.predict(input_data)[0]

        if result == 1:
            message = "Prediction: Loan Approved"
        else:
            message = "Prediction: Loan Not Approved"

        return render_template("index.html", prediction=message)

    except Exception as error:
        return render_template(
            "index.html",
            prediction=f"Error: {error}"
        )


if __name__ == "__main__":
    app.run(debug=True)