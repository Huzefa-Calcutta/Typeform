import os
from flask import Flask, jsonify, request, Response, json
import sklearn, joblib
import pandas as pd
app = Flask(__name__)

# loading the model
model = joblib.load(os.path.join('rf_model.pkl'))


@app.route('/', methods=['GET'])
def test():
    """
    For testing API at http://0.0.0.0:8000/
    :return: Simple json message
    """
    return jsonify({'message': 'Welcome to Typeform'})


@app.route('/predict', methods=["POST"])
def post_rec_list():
    """
    Find best matching chefs/service providers given foodie/consumer's inputs
    Given JSON input parameters like:
    {
        "form_id": "1115313",
        "features": "0.0-0.0-0.0-0.0-0.0-0.0-1.0-0.0-1.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-2.0-0.0-2.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-0.0-2.0-1.0-2.0"
    }
    Return JSON like:
    [
        {
            "form_id": "1115313",
            "submission_view_ratio": "0.654",
        }
    ]
    Address: http://0.0.0.0:8000/form/predict
    """
    message = request.json
    features = [int(x) for x in message['features'].split("-")]
    column_names = ["x_%d" % i for i in range(len(features))]
    submission_view_ratio = model.predict(pd.DataFrame(features, columns=column_names))

    results_json = json.dumps({"form_id":message["form_id"], "submission_view_ratio": submission_view_ratio[0]}, ensure_ascii=False)
    return Response(results_json, mimetype='application/json')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8000, processes=True, use_reloader=True)
