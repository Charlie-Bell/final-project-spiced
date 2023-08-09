from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import re
import html
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.gpt2 import Generator
from src.bert_discriminator import Discriminator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Tokenizers + Models
generator = Generator(MODEL_PATH="./models/gpt2/final")
tokenizer_bert = AutoTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=True) # Need to retrain with do_lower_case=False
discriminator = Discriminator(MODEL_PATH="./models/bert_discriminator/final")
predictor = AutoModelForSequenceClassification.from_pretrained('./models/bert_predictor/final').to(device)

def run_pipeline(input_text):

    realistic_texts = []
    while not realistic_texts:
        texts = generator.inference(input_text)
        realistic_texts = discriminator.discriminate(texts)

    # Predict
    scores = []
    for text in realistic_texts:
        test_input = tokenizer_bert(text, return_tensors='pt').to(device)
        with torch.no_grad():
            output = predictor(**test_input)

        scores.append(output.logits[0][0].cpu().numpy())

    output_text = realistic_texts[np.argmax(scores)]

    response = {
        'comment': html.unescape(input_text),
        'reply': output_text,
    }

    # Result should be response
    print(response)

    return response



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
        response_text = run_pipeline(input_text)
        response = jsonify(response_text)
        print(response)
    return response

if __name__ == "__main__":
    app.run(port=5000, debug=True)