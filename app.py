from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the preprocessor and model from disk
with open(r'C:\Users\Sumeet Maheshwari\Desktop\end to end project\shipmet cost prediction\shipment-cost-prediction\prediction_files/preprocessed.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
    
with open(r'C:\Users\Sumeet Maheshwari\Desktop\end to end project\shipmet cost prediction\shipment-cost-prediction\prediction_files/model.pkl',  'rb') as f:
    model = pickle.load(f)
@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    pack_price = int(request.form['pack-price'])
    unit_price = float(request.form['unit-price'])
    weight_kg = float(request.form['weight-kg'])
    line_item_quantity = int(request.form['line-item-quantity'])
    fulfill_via = request.form['fulfill-via']
    shipment_mode = request.form['shipment-mode']
    country = request.form['country']
    brand = request.form['brand']
    sub_classification = request.form['sub-classification']
    first_line_designation = request.form['first-line-designation']
    
    # Preprocess input values
    X = np.array([[pack_price, unit_price, weight_kg, line_item_quantity, fulfill_via, shipment_mode,
                   country, brand, sub_classification, first_line_designation]])
    X_processed = preprocessor.transform(X.reshape(1,-1))
    
    # Make a prediction
    y_pred = model.predict(X_processed)
    prediction = round(float(y_pred[0]), 2)
    
    return render_template('app.html', prediction=prediction)

if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 5001  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)