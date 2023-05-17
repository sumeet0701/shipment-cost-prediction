from flask import Flask, render_template,request, send_file, redirect,url_for,flash
from flask_cors import CORS, cross_origin
from prediction.batch_prediction import batch_prediction
from prediction.instance_prediction import instance_prediction_class
from shipment_cost_prediction.pipeline.training_pipeline import Pipeline
from shipment_cost_prediction.pipeline.prediction_pipeline import Prediction_Pipeline
from shipment_cost_prediction.constant import *
from shipment_cost_prediction.logger import logging
import shutil


input_file_path = "dataset.csv"
feature_engineering_file_path = "prediction_files/feat_eng.pkl"
transformer_file_path = "prediction_files/preprocessed.pkl"
model_file_path = "prediction_files/model.pkl"

app = Flask(__name__,template_folder = "template")
CORS(app)
app.secret_key = APP_SECRET_KEY

app = Flask(__name__)
CORS(app)
app.secret_key = APP_SECRET_KEY

@app.route("/", methods =["GET"])
@cross_origin()
def home():
    return render_template("result.html")

@app.route("/bulk_predict", methods =["POST"])
@cross_origin()
def bulk_predict():
    try:
        file = request.files.get("files")
        folder = PREDICTION_DATA_SAVING_FOLDER_KEY

        flash("File uploaded!!","success")

        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)

        file.save(os.path.join(folder,file.filename))

        pred = Prediction_Pipeline()
        output_file = pred.initiate_bulk_prediction()
        path = os.path.basename(output_file)

        flash("Prediction File generated!!","success")
        return send_file(output_file,as_attachment=True)

    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))

@app.route("/single_predict", methods =["POST"])
@cross_origin()
def single_predict():
    try:   
        data = {"pack_price" : int(request.form['pack-price']),
                "unit_price" : float(request.form['unit-price']),
                "weight_kg" : float(request.form['weight-kg']),
                "line_item_quantity" : int(request.form['line-item-quantity']),
                "fulfill_via" : request.form['fulfill-via'],
                "shipment_mode" : request.form['shipment-mode'],
                "country" : request.form['country'],
                "brand" : request.form['brand'],
                "sub_classification" : request.form['sub-classification'],
                "first_line_designation" : request.form['first-line-designation']}

        pred = Prediction_Pipeline()
        output = pred.initiate_single_prediction(data)
        flash(f"Predicted Cost for Shipment for given conditions: {output}","success")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))
    

@app.route("/start_train", methods=['GET', 'POST'])
@cross_origin()
def trainRouteClient():
    try:
        train_obj = Pipeline()
        train_obj.run_training_pipeline() # training the model for the files in the table
    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))


if __name__=="__main__":
    
    port = int(os.getenv("PORT",5000))
    host = '0.0.0.0'
    app.run(host=host,port=port,debug=True)