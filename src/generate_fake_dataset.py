from src.gpt2 import Generator
import pandas as pd


generator = Generator(MODEL_PATH="./models/gpt2/final")
generator.tokenizer.padding_side = "left"

train = pd.read_csv("./data/preprocessed/train.csv", index_col='Unnamed: 0')[['text']]
validation = pd.read_csv("./data/preprocessed/validation.csv", index_col='Unnamed: 0')[['text']]

train['fake'] = False
validation['fake'] = False

train_texts = train['text'].values.tolist()
validation_texts = validation['text'].values.tolist()


# Function to create fakes for a batch of input texts
def create_fakes(input_texts):
    MAX_NEW_TOKENS = 40
    prompt_texts = [input_text.split(generator.SEP_TOKEN)[0]+generator.SEP_TOKEN for input_text in input_texts]       
    model_inputs = generator.tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=1024-MAX_NEW_TOKENS).to(generator.torch_device)

    sample_outputs = generator.model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        early_stopping=True,
        top_k=50,
        top_p=0.8,
        temperature=0.95,
        pad_token_id=generator.tokenizer.pad_token_id,
    )
    generated_texts = generator.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
    del model_inputs, sample_outputs
    return generated_texts


# Split list into batches (list of lists)
def batch_generations(texts, batch_size):
    for i in range(0, len(texts), batch_size): 
        yield texts[i:i + batch_size]

BATCH_SIZE = 5
num_train_fakes = int(len(train_texts)/2)
num_validation_fakes = int(len(validation_texts)/2)
train_batches = list(batch_generations(train_texts[:num_train_fakes], BATCH_SIZE))
validation_batches = list(batch_generations(validation_texts[:num_validation_fakes], BATCH_SIZE))


# Function to batch generate fakes
def batch_generate_fakes(batches):
    fakes_dict = {
        'text': [],
        'fake': True,
    }

    num_batches = len(batches)
    for i, batch in enumerate(batches):
        fake_batch = create_fakes(batch)
        if (i%50==0):
            print(f"Batch {i} of {num_batches}")
            print(fake_batch)  
        fakes_dict['text'].extend(fake_batch)

    return fakes_dict


# First half of validation set is replaced with fakes
validation_fakes = pd.DataFrame(batch_generate_fakes(validation_batches))
validation.iloc[0:len(validation_fakes)] = validation_fakes
validation.to_csv(generator.DATA_PROC_DIR + "/validation_fakes.csv")

# First half of train set is replaced with fakes.
train_fakes = pd.DataFrame(batch_generate_fakes(train_batches))
train.iloc[0:len(train_fakes)] = train_fakes
train.to_csv(generator.DATA_PROC_DIR + "/train_fakes.csv")