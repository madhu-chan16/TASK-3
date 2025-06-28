from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    result = f"Predicted House Price: ${prediction[0] * 100000:.2f}"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
