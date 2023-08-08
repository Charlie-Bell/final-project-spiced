import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch import cuda, no_grad
import re
import html
from datetime import datetime


class Discriminator():
    def __init__(self,
                 MODEL_DIR="models/bert_discriminator",
                 MODEL_PATH="distilbert-base-cased",
                 DATA_RAW_DIR="./data/raw/",
                 DATA_PROC_DIR="./data/preprocessed/",
                 EOS_TOKEN='<|endoftext|>',
                 SEP_TOKEN='<\|reply\|>',
                 MAX_LENGTH=512,
                 TRAIN_RATIO=0.9,
                 BATCH_SIZE=4,
                 EPOCHS=1,
                 LEARNING_RATE=2e-5,
                 SEED=42
                 ):
              
        # Settings
        self.MODEL_DIR = MODEL_DIR
        self.MODEL_PATH = MODEL_PATH
        self.DATA_RAW_DIR = DATA_RAW_DIR
        self.DATA_PROC_DIR = DATA_PROC_DIR
        self.EOS_TOKEN = EOS_TOKEN
        self.SEP_TOKEN = SEP_TOKEN
        self.MAX_LENGTH = MAX_LENGTH
        self.TRAIN_RATIO = TRAIN_RATIO
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.SEED = SEED

        # Device
        self.torch_device = "cuda" if cuda.is_available() else "cpu"
        print("Using device: " + self.torch_device)

        # Tokenizer + Model
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=True) # !!!!need to retrain with do_lower_case=False
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_PATH).to(self.torch_device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def regex_text(self, text):
        text = html.unescape(text)
        text = re.sub(r"\\'", r"'", text)
        text = re.sub(r"\s+$", '', text)    
        if type(text)==str:
            return text
        return re.findall(self.SEP_TOKEN + " (,?.*)", text)

    def label_to_list(self, label):
        if label:
            return [1]
        else:
            return [0]

    def clean_dataframe(self, df):
        df = df[df['text'].str.contains(self.sep_token)]
        df['text'] = df['text'].apply(self.regex_text)
        df = df[df['text'].str.len() != 0]
        df['text'] = df['text'].apply(lambda x: x[0])
        df['label'] = df['label'].apply(self.label_to_list)
        return df
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True, max_length=self.MAX_LENGTH)

    def preprocessing(self):
        ### Preprocessing
        train = pd.read_csv(self.DATA_PROC_DIR + "/train_fakes.csv", index_col=0, encoding='utf-8', engine='python')
        validation = pd.read_csv(self.DATA_PROC_DIR + "/validation_fakes.csv", index_col=0, encoding='utf-8', engine='python')

        validation = self.clean_dataframe(validation)
        train = self.clean_dataframe(train)

        dataset = dict()
        dataset['validation'] = Dataset.from_pandas(validation, preserve_index=False)
        dataset['train'] = Dataset.from_pandas(train, preserve_index=False)
        datasets = DatasetDict(dataset)

        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            )

        return tokenized_datasets

    def train_model(self, dataset, SAVE_STEPS=10000, model_name=None):
        training_args = TrainingArguments(
            output_dir=self.MODEL_DIR,
            evaluation_strategy="epoch",
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.01,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            num_train_epochs=self.EPOCHS,
            save_steps=SAVE_STEPS,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=self.data_collator,
        )

        trainer.train()

        if model_name:
            trainer.save_model(self.MODEL_DIR + "/" + model_name)
        else:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
            trainer.save_model(self.MODEL_DIR + "/model-" + dt_string)

        
    def run_training_pipeline(self):
        dataset = self.preprocessing()
        self.train_model(dataset)

    def discriminate(self, texts):
        realistic_texts = []
        texts = [self.regex_text(text) for text in texts[:]]
        for text in texts:
            test_input = self.tokenizer(text, return_tensors='pt').to(self.torch_device)
            with no_grad():
                logits = self.model(**test_input).logits

            predicted_class_id = logits.argmax().item()

            if not predicted_class_id:
                realistic_texts.append(text)

        return realistic_texts