import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch import cuda, no_grad
import re
import html
from datetime import datetime


class Predictor():
    def __init__(self,
                 MODEL_DIR="models/bert_predictor",
                 MODEL_PATH="distilbert-base-cased",
                 DATA_RAW_DIR="./data/raw/",
                 DATA_PROC_DIR="./data/preprocessed/",
                 EOS_TOKEN='<|endoftext|>',
                 SEP_TOKEN='<|reply|>',
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
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_PATH, num_labels=1).to(self.torch_device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def regex_text(self, text):
        text = html.unescape(text)
        text = re.sub(r"\\'", r"'", text)
        text = re.sub(r"\s+$", '', text)  
        texts = re.findall(self.SEP_TOKEN + " (,?.*)", text)
        for t in texts:
            if t:
                text = t
                break
        text = text.rstrip()
        return text
    
    def clean_dataframe(self, df):
        df['completion'] = df['completion'].astype(str)
        df['completion'] = df['completion'].apply(self.regex_text)
        df = df[df['completion'].str.len() != 0]
        return df
    
    def minmax_scale(self, X, X_min, X_max):
        X_scaled = (X - X_min) / (X_max - X_min)
        return X_scaled

    # Scaling is [MinMax -> np.exp -> MinMax] such that the comments/replies with a higher count have more influence
    # To do: Fit scale to training data only
    def scale(self, df, cols=['comment_score', 'reply_score']):
        min_score = df[cols].min().min()
        max_score = df[cols].max().max()
        for col in cols:
            df[col+"_scaled"] = df[col]
            col = col+"_scaled"
            df[col] = df[col].apply(self.minmax_scale, args=(min_score, max_score))
            df[col] = df[col].apply(np.exp)

        cols = ['comment_score_scaled', 'reply_score_scaled']
        min_score = df[cols].min().min()
        max_score = df[cols].max().max()
        for col in cols:
            df[col] = df[col].apply(self.minmax_scale, args=(min_score, max_score))
        return df
    
    def scale_dataframe(self, df):
        df = self.scale(df)
        df['reply_score_minmax'] = df['reply_score'].apply(self.minmax_scale, args=(df['reply_score'].min(), df['reply_score'].max()))
        df['score_ratio'] = df['reply_score_scaled']/df['comment_score_scaled']
        return df
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["completion"], truncation=True, padding=True, max_length=self.MAX_LENGTH)
    
    def preprocessing(self):
        validation = pd.read_csv(self.DATA_PROC_DIR + '/validation.csv', index_col=0)[['completion', 'comment_score', 'reply_score']]
        validation = self.clean_dataframe(validation)
        validation = self.scale_dataframe(validation)
        validation = validation[['completion', 'reply_score_minmax']].rename(columns={'reply_score_minmax': 'label'})

        train = pd.read_csv(self.DATA_PROC_DIR + '/train.csv', index_col=0)[['completion', 'comment_score', 'reply_score']]
        train = self.clean_dataframe(train)
        train = self.scale_dataframe(train)
        train = train[['completion', 'reply_score_minmax']].rename(columns={'reply_score_minmax': 'label'})

        dataset = dict()
        dataset['validation'] = Dataset.from_pandas(validation, preserve_index=False)
        dataset['train'] = Dataset.from_pandas(train, preserve_index=False)
        datasets = DatasetDict(dataset)

        datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["completion"],
            )

        return datasets
    
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

    def predict(self, realistic_texts):
        scores = []
        for i, text in enumerate(realistic_texts):
            test_input = self.tokenizer(text, return_tensors='pt').to(self.torch_device)
            with no_grad():
                output = self.model(**test_input)

            scores.append(output.logits[0][0].cpu().numpy())

        output_text = realistic_texts[np.argmax(scores)]
        return output_text