from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load ML model (handle both local and Vercel paths)
try:
    if os.path.exists('house_price_model.pkl'):
        model = joblib.load('house_price_model.pkl')
    else:
        # If model file doesn't exist, handle gracefully
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            if model is None:
                prediction = 'Error: Model not loaded'
            else:
                # Get input values from the form
                feature1 = float(request.form['feature1'])  # OverallQual
                feature2 = float(request.form['feature2'])  # GrLivArea
                feature3 = float(request.form['feature3'])  # GarageCars
                feature4 = float(request.form['feature4'])  # TotalBsmtSF

                input_data = np.array(
                    [[feature1, feature2, feature3, feature4]])

                # Predict the output
                result = model.predict(input_data)[0]
                prediction = round(result, 2)

        except Exception as e:
            prediction = f'Error: {e}'

    return render_template("index.html", prediction=prediction)


# For Vercel deployment
if __name__ == '__main__':
    app.run(debug=False)
