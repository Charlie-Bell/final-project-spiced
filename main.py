from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import numpy as np
import re
import html
from flask import Flask, request, jsonify


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Tokenizers + Models
tokenizer_gpt = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer_bert = AutoTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=True) # Need to retrain with do_lower_case=False
generator = AutoModelForCausalLM.from_pretrained('./models/gpt2/final', pad_token_id=tokenizer_gpt.eos_token_id).to(device)
discriminator = AutoModelForSequenceClassification.from_pretrained('./models/bert_discriminator/final').to(device)
predictor = AutoModelForSequenceClassification.from_pretrained('./models/bert_predictor/final').to(device)

def regex_text(text):
            text = html.unescape(text)
            text = re.sub(r"\\'", r"'", text)
            text = re.sub(r"\s+$", '', text)    
            return text

def run_pipeline(input_text):
    # Generate
    sep_token = "<|reply|>"
    model_inputs = tokenizer_gpt([" ".join([input_text, sep_token])], return_tensors='pt').to(device)


    texts = []
    realistic_texts = []
    while not realistic_texts:
        # Generator 
        sample_outputs = generator.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=True,
        early_stopping=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=20,
        )   
         
        for i, sample_output in enumerate(sample_outputs):
            text = tokenizer_gpt.decode(sample_output, skip_special_tokens=False).split('<|reply|>')[1].split('\n')[0][1:]
            texts.append(text)

        # Discriminate
        texts = [regex_text(text) for text in texts[:]]
        for text in texts:
            test_input = tokenizer_bert(text, return_tensors='pt').to(device)
            with torch.no_grad():
                logits = discriminator(**test_input).logits

            predicted_class_id = logits.argmax().item()
            if not predicted_class_id:
                realistic_texts.append(text)

    # Predict
    scores = []
    for text in realistic_texts:
        test_input = tokenizer_bert(text, return_tensors='pt').to(device)
        with torch.no_grad():
            output = predictor(**test_input)

        scores.append(output.logits[0][0].cpu().numpy())

    output_text = realistic_texts[np.argmax(scores)]

    response = {
        'comment': input_text,
        'reply': output_text,
    }

    # Result should be response
    print(response)

    return response


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
        input_text = data_json['input_text']
        print(input_text)
        response_text = run_pipeline(input_text)
        response = jsonify({'response_text': response_text})
        print(response)
    return response

if __name__ == "__main__":
    app.run(port=5000, debug=True)