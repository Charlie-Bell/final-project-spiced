import html
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.gpt2 import Generator
from src.bert_discriminator import Discriminator
from src.bert_predictor import Predictor


# --- Generator -> Discriminator -> Predictor --- #
generator = Generator(MODEL_PATH="./models/gpt2/final")
discriminator = Discriminator(MODEL_PATH="./models/bert_discriminator/final")
predictor = Predictor(MODEL_PATH="./models/bert_predictor/final")

def inference_pipeline(input_text):
    realistic_texts = []
    while not realistic_texts:
        candidate_texts = generator.generate_candidates(input_text)
        realistic_texts = discriminator.discriminate(candidate_texts)

    output_text = predictor.predict(realistic_texts)

    response = {
        'comment': html.unescape(input_text),
        'reply': output_text,
    }

    print(response)
    return response

# --- Flask App --- #
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
        response_text = inference_pipeline(input_text)
        response = jsonify(response_text)
        print(response)
    return response

if __name__ == "__main__":
    app.run(port=5000, debug=True)