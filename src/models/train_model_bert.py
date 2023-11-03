import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup

from transformers import BertTokenizer, BertModel

from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

PATH = "../../data/inheritim/"
PATH_OUT = "../../models/bert-detoxification/"
BATCH_SIZE = 32
EPOCH_NUM = 1
lr = 2e-5
bert_model_name = 'bert-base-cased'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def preprocess_dataset():
    """
    Read dataset and prepare it
    :return: train_set, valid_set
    """
    train_df = pd.read_csv(PATH + 'jigsaw.csv', index_col=0)
    df_train, df_valid = train_test_split(train_df, test_size=0.2, stratify=train_df["toxic"], random_state=SEED)
    df_train = pd.concat(
        [
            df_train[df_train["toxic"] == 0].sample(
                10000, random_state=SEED, replace=False
            ),
            df_train[df_train["toxic"] == 1].sample(
                10000, random_state=SEED, replace=False
            ),
        ]
    )
    df_valid = pd.concat(
        [
            df_valid[df_valid["toxic"] == 0].sample(
                1000, random_state=SEED, replace=False
            ),
            df_valid[df_valid["toxic"] == 1].sample(
                1000, random_state=SEED, replace=False
            ),
        ]
    )
    return df_train, df_valid


def clear_str(string: str) -> str:
    """
    Tokenization/string cleaning for text
    """
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    string = re.sub(r"([?.!,¿])", r" \1 ", string)
    string = re.sub(r'[" "]+', " ", string)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    string = re.sub(r"[^a-zA-Z?.!,¿]+", " ", string)
    return string.strip().lower()


class ToxicDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, dataframe: pd.DataFrame):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.df = dataframe

    def row_to_tensor(self, tokenizer: BertTokenizer, row: pd.Series, max_len: int=120):
        tokens = tokenizer.encode(clear_str(row["comment_text"]), add_special_tokens=True, max_length=max_len, truncation=True)
        x = torch.LongTensor(tokens)

        labels = ['non_toxic', 'toxic']
        y = torch.FloatTensor(row[labels])
        return x, y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        return self.row_to_tensor(self.tokenizer, self.df.iloc[index])


def collate_fn(batch: torch.Tensor, device: torch.device) -> (torch.Tensor, torch.Tensor):
    """
    merges a list of samples to form a mini-batch of Tensors
    :param batch: batch of data
    :param device: cpu or gpu
    :return:
    """
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x.to(device), y.to(device)


class BertClassifier(nn.Module):
    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None):
        x = self.bert(input_ids, attention_mask=attention_mask)
        cls_x = x[1]
        cls_x = self.classifier(cls_x)
        out = self.softmax(cls_x)
        return out


def train(model: BertClassifier, iterator, optimizer, scheduler, loss_func):
    """
    Train loop
    :param model: BertClassifier
    :param iterator: our dataloader
    :param optimizer: our optimizer
    :param scheduler: our sheduler
    :param loss_func: our loss function
    :return: None
    """
    model.train()
    total_loss = 0

    iteration = 0
    for x, y in tqdm(iterator):
        iteration += 1
        optimizer.zero_grad()
        mask = (x != 0).float()
        outputs = model(x, attention_mask=mask)
        loss = loss_func(outputs, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Train loss {total_loss / len(iterator)}\n")


def evaluate(model, iterator, loss_func, verbose=True):
    """
    Evaluation loop
    :param model: BertClassifier
    :param iterator: our dataloader
    :param loss_func: our loss function
    :param verbose: print loss and ROC_AUC score if True
    :return: (true values, prediction)
    """
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in iterator:
            mask = (x != 0).float()
            outputs = model(x, attention_mask=mask)
            loss = loss_func(outputs, y)
            total_loss += loss
            outputs = torch.argmax(outputs, dim=1)
            y = torch.argmax(y, dim=1)
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)

    labels = ['toxic']
    if verbose:
        print("\nROC_AUC for labels:")
        for i, name in enumerate(labels):
            print(f" * {name} - {roc_auc_score(true, pred)}")
        print(f"\nEVALUATE LOSS -  {total_loss / len(iterator)}\n")

    return true, pred


if __name__ == "__main__":
    print("Start! Training...")

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    df_train, df_valid = preprocess_dataset()
    train_dataset = ToxicDataset(tokenizer, df_train)
    val_dataset = ToxicDataset(tokenizer, df_valid)

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(val_dataset)

    collate_fn = partial(collate_fn, device=device)
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                sampler=train_sampler, collate_fn=collate_fn)

    val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              sampler=dev_sampler, collate_fn=collate_fn)

    bert = BertModel.from_pretrained(bert_model_name)
    model = BertClassifier(bert, 2).to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # The triangle learning rate advances linearly until half of the first epoch, then linearly decays.
    w_steps = 10 ** 3
    t_steps = len(train_iterator) * EPOCH_NUM - w_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, w_steps, t_steps)

    torch.cuda.empty_cache()

    for i in range(EPOCH_NUM):
        print('=' * 50, f"EPOCH {i + 1}", '=' * 50)
        print("\nTRAINING\n")
        train(model, train_iterator, optimizer, scheduler, loss_func)
        print("\nEVALUATING\n")
        evaluate(model, val_iterator, loss_func)

    print("Please, wait, saving checkpoint!...")
    torch.save(model.state_dict(), PATH_OUT + "model-final.pt")
    print("Done! Model checkpoint saved!")
