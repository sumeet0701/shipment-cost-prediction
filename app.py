from flask import Flask, render_template, request
from prediction.batch_prediction import batch_prediction
from prediction.instance_prediction import instance_prediction_class
from werkzeug.utils import secure_filename
import os
from shipment_cost_prediction.constant import * 

input_file_path = "SCMS_Delivery_History_Dataset.csv"
feature_engineering_file_path = "prediction_files/feat_eng.pkl"
transformer_file_path = "prediction_files/preprocessed.pkl"
model_file_path = "prediction_files/model.pkl"




COUNTRY_MAP = {'Zambia': 0, 'Ethiopia': 1, 'Nigeria': 2, 'Tanzania': 3, "Côte d'Ivoire": 4, 'Mozambique': 5, 'Others': 6, 'Zimbabwe': 7, 'South Africa': 8, 'Rwanda': 9, 'Haiti': 10, 'Vietnam': 11, 'Uganda': 12}
FULFILL_VIA_MAP = {'From RDC': 0, 'Direct Drop': 1}
SHIPMENT_MODE_MAP = {'Truck': 0, 'Air': 1, 'Air Charter': 2, 'Ocean': 3}
SUB_CLASSIFICATION_MAP = {'Adult': 0, 'Pediatric': 1, 'HIV test': 2, 'HIV test - Ancillary': 3, 'Malaria': 4, 'ACT': 5}
DOSAGE_FORM= {'Tablet': 0, 'Test kit': 1, 'Oral': 2, 'Capsule': 3}
BRAND_MAP = {'Generic': 0, 'Others': 1, 'Determine': 2, 'Uni-Gold': 3}
#FIRST_LINE_DESIGNATION_MAP = {'Yes': 0, 'No': 1}

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'

app = Flask(__name__,template_folder='template')
ALLOWED_EXTENSIONS = {'csv'}


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/batch", methods=["POST"])
def perform_batch_prediction():

    file = request.files['csv_file']  # Update the key to 'csv_file'
    # Check if the file has a valid extension
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        # Delete all files in the file path
        for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
            file_path = os.path.join(UPLOAD_FOLDER , filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save the new file to the uploads directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(file_path)

        # Perform batch prediction using the uploaded file
        batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
        batch.start_batch_prediction()


        output = "Batch Prediction Done"
        return render_template("index.html", prediction_result=output, prediction_type='batch')
    else:
        return render_template('index.html', prediction_type='batch', error='Invalid file type')


@app.route("/instance", methods=["POST"])
def perform_instance_prediction():
    weight_kg = float(request.form['weight_kg'])
    line_item_quantity = int(request.form['line_item_quantity'])
    line_item_value=float(request.form['line_item_value'])
    fulfill_via = request.form['fulfill_via']
    shipment_mode = request.form['shipment_mode']
    country = request.form['country']
    brand = request.form['brand']
    sub_classification = request.form['sub_classification']
    dosage_form=request.form['dosage_form']

    predictor = instance_prediction_class(weight_kg, line_item_quantity,line_item_value,
                                                            fulfill_via, shipment_mode, country, brand,
                                                            sub_classification,dosage_form)
    predicted_price = predictor.predict_price_from_input()

    return render_template('index.html', prediction_type='instance', predicted_price=predicted_price)


if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8000  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)