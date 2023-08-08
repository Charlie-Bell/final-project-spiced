from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from datetime import datetime
import pandas as pd
import torch
import os
import html

class Generator:
    def __init__(self,
                MODEL_DIR="models/gpt2",
                MODEL_PATH="distilgpt2",
                DATA_RAW_DIR="./data/raw/",
                DATA_PROC_DIR="./data/preprocessed/",
                EOS_TOKEN='<|endoftext|>',
                SEP_TOKEN='<|reply|>',
                MAX_LENGTH=512,
                TRAIN_RATIO=0.9,
                BATCH_SIZE=2,
                EPOCHS=2,
                SEED=42):
        
        """
        Can load model for training e.g.
        generator = Generator(MODEL_PATH='distilgpt2')
        generator.run_training_pipeline()
        or
        Can load model for inference e.g.
        generator = Generator(MODEL_PATH='./path/to/trained/model')
        generator.inference(input_text)
        """

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
        self.SEED = SEED

        # Device
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: " + self.torch_device)

        # Tokenizer + Model + Data Collator
        self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2', return_tensors='pt', eos_token=EOS_TOKEN, pad_token=EOS_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH).to(self.torch_device)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)


    # Functions
    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=self.MAX_LENGTH)

    def label_function(self, examples):
        examples['labels'] = examples['input_ids']
        return examples

    def insert_tokens(self, pair):
        return " ".join([self.EOS_TOKEN, str(pair[0]), self.SEP_TOKEN, str(pair[1]), self.EOS_TOKEN])

    def preprocess_csvs(self, DATA_RAW_DIR=None):
        # Load and combine dataframes
        if DATA_RAW_DIR:
            self.DATA_RAW_DIR = DATA_RAW_DIR
        filenames = os.listdir(self.DATA_RAW_DIR)
        dfs = [pd.read_csv(self.DATA_RAW_DIR + name, index_col='Unnamed: 0') for name in filenames]
        df = pd.concat(dfs)
        df = df[['comment', 'reply', 'comment_score', 'reply_score']]
        df = df.rename(columns={'comment': 'prompt', 'reply': 'completion'})
        df[['prompt', 'completion']] = df[['prompt', 'completion']].applymap(str)
        df[['prompt', 'completion']] = df[['prompt', 'completion']].applymap(html.unescape)
        df = df.reset_index(drop=True, inplace=False)
        df = df.sample(frac=1, random_state=self.SEED)
        df = df.reset_index(drop=True, inplace=False)
        df.to_csv(self.DATA_PROC_DIR+"gpt2_preprocessed.csv")

    def create_dataset_dict(self, df=None, DATA_PATH=None):
        # If user wants to use a different dataset
        if not df:
            if DATA_PATH:
                df = pd.read_csv(DATA_PATH)
            else:
                df = pd.read_csv(self.DATA_PROC_DIR+"gpt2_preprocessed.csv")


        # Stringify
        prompts = df['prompt'].to_list()
        completions = df['completion'].to_list()

        # Insert Tokens
        texts = [self.insert_tokens(pair) for pair in zip(prompts, completions)]

        # Define split
        texts_size = len(texts)
        train_size = int(self.TRAIN_RATIO*texts_size)

        # Split
        texts_train = texts[:train_size]
        texts_validation = texts[train_size:]

        # Convert to DatasetDict
        dataset = dict()
        dataset['train'] = Dataset.from_dict({'text': texts_train})
        dataset['validation'] = Dataset.from_dict({'text': texts_validation})
        dataset_dict = DatasetDict(dataset)
        return dataset_dict
    
    def preprocess_dataset_dict(self, dataset_dict, save_dataset=False):
        tokenized_dataset = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=['text'],
            )
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask'], output_all_columns=True)

        gpt2_dataset = tokenized_dataset.map(
            self.label_function,
            batched=True,
            num_proc=1,
            )
        gpt2_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
        
        if save_dataset:
            gpt2_dataset.save_to_disk(self.DATA_PROC_DIR + "/dataset")

        return gpt2_dataset

    def train_model(self, dataset, SAVE_STEPS=10000, model_name=None):
        training_args = TrainingArguments(
            output_dir=self.MODEL_DIR,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
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
        self.preprocess_csvs()
        dataset = self.create_dataset_dict()
        dataset = self.preprocess_dataset_dict(dataset)
        self.train_model(dataset)

    def inference(self, input_text):
        # Convert to regular text
        input_text = html.unescape(input_text)

        # Generate
        model_inputs = self.tokenizer([" ".join([input_text, self.SEP_TOKEN])], return_tensors='pt').to(self.torch_device)

        texts = []

        # Generator 
        sample_outputs = self.model.generate(
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
            text = self.tokenizer.decode(sample_output, skip_special_tokens=False).split(self.SEP_TOKEN)[1].split('\n')[0][1:]
            texts.append(text)
            print(text)

        return texts