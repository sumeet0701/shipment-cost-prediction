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
        output_file = pred.initiate_bulk_predictions()
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
        data = {"Pack_Price" : int(request.form['pack-price']),
                "Unit_Price" : float(request.form['unit-price']),
                "Weight_Kilograms_Clean" : float(request.form['weight-kg']),
                "Line_Item_Quantity" : int(request.form['line-item-quantity']),
                "Fulfill_Via" : request.form['fulfill-via'],
                "Shipment_Mode" : request.form['shipment-mode'],
                "Country" : request.form['country'],
                "Brand" : request.form['brand'],
                "Sub_Classification" : request.form['sub-classification'],
                "First_Line_Designation" : request.form['first-line-designation']}

        pred = instance_prediction_class()
        preprocess = pred.preprocess_input(data=data)          
        output = pred.predict_price(preprocess)
        #output = pred.preprocess_input(data)
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