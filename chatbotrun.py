# Part 1
import argparse
import re
from time import time

# Upgrade torchao to a compatible version
!pip install --upgrade torchao

from nltk.metrics import agreement
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = (
    "You are a helpful customer support agent responding to customers on Twitter. "
    "Keep replies concise, polite, and on-topic."
)

# Part 2 
class chatbot:

    def __init__(self, args=None):
        parser = argparse.ArgumentParser(description='Fine-tune / run a small LLM as a support chatbot')

        # Data / IO
        parser.add_argument('--data_path', help='Path to the training/inference CSV')
        parser.add_argument('--outpath', help='Output directory (must end with /)')
        parser.add_argument('--mode', help='train/inference', default='train')
        parser.add_argument('--version', help='version tag for output files', default='v1')
        parser.add_argument('--num_train_records', help='rows to use for training', type=int, default=20000)

        # Model
        parser.add_argument('--base_model', help='HF hub id or local path of the base model',
                             default='Qwen/Qwen2.5-0.5B-Instruct')
        parser.add_argument('--load_model_from', help='path to a saved LoRA adapter (inference mode)')
        parser.add_argument('--max_seq_len', help='max tokens per example', type=int, default=256)

        # LoRA
        parser.add_argument('--lora_r', type=int, default=16)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=float, default=0.05)

        # Training
        parser.add_argument('--epochs', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=2e-4)

        if args is None:
        # Running from command line
         args, _ = parser.parse_known_args()

        self.data_path = args.data_path
        self.outpath = args.outpath
        self.mode = args.mode
        self.version = args.version
        self.num_train_records = args.num_train_records

        self.base_model_name = args.base_model
        self.load_model_from = args.load_model_from
        self.max_seq_len = args.max_seq_len

        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Data loading (carried over from the original script; the dataset's
    # shape hasn't changed, only what we do with it)
    # ------------------------------------------------------------------
    def process_data(self, path):
        print("Data path is:", path)

    # Now try to read the CSV
        data = pd.read_csv(self.data_path)


        if self.mode == 'train':
            data['in_response_to_tweet_id'] = data['in_response_to_tweet_id'].fillna(-12345)
            tweets_in = data[data['in_response_to_tweet_id'] == -12345]
            tweets_in_out = tweets_in.merge(data, left_on=['tweet_id'], right_on=['in_response_to_tweet_id'])
            return tweets_in_out[:self.num_train_records]
        elif self.mode == 'inference':
            return data

    def replace_anonymized_names(self, data):

        def replace_name(match):
            cname = match.group(2).lower()
            if not cname.isnumeric():
                return match.group(1) + match.group(2)
            return '@__cname__'

        re_pattern = re.compile(r'(\W@|^@)([a-zA-Z0-9_]+)')
        if self.mode == 'train':
            in_text = data['text_x'].apply(lambda txt: re_pattern.sub(replace_name, txt))
            out_text = data['text_y'].apply(lambda txt: re_pattern.sub(replace_name, txt))
            return list(in_text.values), list(out_text.values)
        else:
            return list(map(lambda x: re_pattern.sub(replace_name, x), data))

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------
    def build_chat_text(self, tokenizer, customer_text, support_text=None):
        """Builds a chat-formatted training example using the base model's
        own chat template, so the model sees text in the exact format it
        was originally instruction-tuned on."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": customer_text},
        ]
        if support_text is not None:
            messages.append({"role": "assistant", "content": support_text})
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def build_dataset(self, tokenizer, in_text, out_text):
        texts = [self.build_chat_text(tokenizer, i, o) for i, o in zip(in_text, out_text)]
        ds = Dataset.from_dict({"text": texts})

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
            )

        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        return ds

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def load_base_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
        ).to(self.device)

        return model, tokenizer

    def attach_lora(self, model):
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------
    # Train / generate
    # ------------------------------------------------------------------
    def train_model(self, model, tokenizer, train_dataset):
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir = os.path.join(self.outpath, "checkpoints"),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=20,
            save_strategy="epoch",
            fp16=(self.device == "cuda"),
            bf16=False,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        trainer.train()

        adapter_path = os.path.join(self.outpath, "adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        return model

    def generate_response(self, model, tokenizer, sentences):
        model.eval()
        output_responses = []
        for sent in sentences:
            prompt = self.build_chat_text(tokenizer, sent, support_text=None)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = out_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            output_responses.append(response)

        out_df = pd.DataFrame()
        out_df['Tweet in'] = sentences
        out_df['Tweet out'] = output_responses
        return out_df

    # ------------------------------------------------------------------
    def main(self):
        if self.mode == 'train':
            data = self.process_data(self.data_path)
            in_text, out_text = self.replace_anonymized_names(data)

            import numpy as np
            test_indexes = np.random.randint(0, len(in_text), min(10, len(in_text)))
            test_sentences = [in_text[i] for i in test_indexes]

            model, tokenizer = self.load_base_model_and_tokenizer()
            model = self.attach_lora(model)

            train_dataset = self.build_dataset(tokenizer, in_text, out_text)
            print(f"Training examples: {len(train_dataset)}")

            model = self.train_model(model, tokenizer, train_dataset)

            test_responses = self.generate_response(model, tokenizer, test_sentences)
            print(test_responses)
            test_responses.to_csv(
    os.path.join(self.outpath, "output_response.csv"),
    index=False,
)

        elif self.mode == 'inference':
            base_model, tokenizer = self.load_base_model_and_tokenizer()
            model = PeftModel.from_pretrained(base_model, self.load_model_from).to(self.device)

            data = self.process_data(self.data_path)
            col = data.columns.tolist()[0]
            test_sentences = list(data[col].values)
            test_sentences = self.replace_anonymized_names(test_sentences)

            responses = self.generate_response(model, tokenizer, test_sentences)
            print(responses)
            responses.to_csv(self.outpath + 'responses_' + str(self.version) + '_.csv', index=False)


# Part 3 
from types import SimpleNamespace

args = SimpleNamespace(
    data_path="/content/twcs.csv",          # changed to twcs.csv
    outpath="/content/output_chatbot/",             # changed output path
    mode="train",
    version="v1",
    num_train_records=20000,

    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    load_model_from=None,
    max_seq_len=256,

    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
)

import os
os.makedirs(args.outpath, exist_ok=True)

start_time = time()

bot = chatbot(args)
bot.main()

end_time = time()

print(f"Processing finished in {end_time-start_time:.2f} seconds")

# Part 4

from peft import PeftModel

# Load base model
base_model, tokenizer = bot.load_base_model_and_tokenizer()

# Load your trained adapter
model = PeftModel.from_pretrained(
    base_model,
    "/content/output_chatbot/adapter"
).to(bot.device)

model.eval()

# Part 5

while True:
    message = input("You: ")

    if message.lower() in ["exit", "quit"]:
        break

    reply = bot.generate_response(model, tokenizer, [message])

    print("Bot:", reply["Tweet out"][0])