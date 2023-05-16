from flask import Flask, render_template, request, jsonify
from prediction.batch_prediction import batch_prediction
from prediction.instance_prediction import instance_prediction_class

input_file_path = "SCMS_Delivery_History_Dataset.csv"
feature_engineering_file_path = "prediction_files/feat_eng.pkl"
transformer_file_path = "prediction_files/preprocessed.pkl"
model_file_path = "prediction_files/model.pkl"

app = Flask(__name__,template_folder='template')

@app.route("/", methods=["GET"])
def home():
    return render_template("result.html")


@app.route('/templates/index.html', methods=['POST'])
def perform_batch_prediction():
    # Perform batch prediction using the batch_prediction_function
    batch_prediction.start_batch_prediction(input_file_path, model_file_path, transformer_file_path,
                                            feature_engineering_file_path)

    # Return the prediction result as a response
    return "Batch Prediction Done"

@app.route('/instance-prediction', methods=['POST'])
def perform_instance_prediction():
    # Perform instance prediction using the instance_prediction_function
    result = instance_prediction_class.predict_price_from_input()

    # Return the prediction result as JSON response
    return jsonify(result)



if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 5000  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)
    
    