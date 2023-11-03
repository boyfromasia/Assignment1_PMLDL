import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from torch import cuda

from transformers import set_seed

SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = 'cuda' if cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PATH = "../../data/inheritim/"
SAVE_PATH = "../../models/t5-detoxification/"
model_checkpoint = "t5-small"
batch_size = 32


class Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.x = tokenizer(df.reference.tolist(),
                           truncation=True,
                           )

        self.y = tokenizer(df.translation.tolist(),
                           truncation=True,
                           )

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.x["input_ids"][idx],
            "attention_mask": self.x["attention_mask"][idx],
            "labels": self.y['input_ids'][idx],
        }

    def __len__(self):
        return len(self.x['input_ids'])


if __name__ == "__main__":
    print("Start! Training...")

    df = pd.read_csv(PATH + 'filtered.csv', index_col=0)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_dataset = Dataset(df_train, tokenizer)
    test_dataset = Dataset(df_test, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint,).to(device)
    args = Seq2SeqTrainingArguments(
        output_dir=SAVE_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=300,
        weight_decay=0.01,
        learning_rate=3e-5,
        logging_steps=1000,
        eval_steps=1000,
        evaluation_strategy='steps',
        save_total_limit=1,
        save_steps=1000,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Please, wait, saving checkpoint!...")
    trainer.save_model(SAVE_PATH + "checkpoint-final/")
    print("Done! Model checkpoint saved!")
