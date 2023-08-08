from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/flask', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        print("GET request")
    if request.method == 'POST':
        data_json = request.get_json()
        print(data_json)
        input_text = data_json['comment']
        print(input_text)
        response_text = run_inference(input_text)
        response = jsonify(response_text)
        print(response)
    return response

app.run(port=5000, debug=True)