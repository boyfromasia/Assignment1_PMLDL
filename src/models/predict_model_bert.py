import re
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

PATH_OUT = "../../models/bert-detoxification/"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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


class BertClassifier(nn.Module):
    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None):
        x = self.bert(input_ids, attention_mask=attention_mask)
        cls_x = x[1] # sentence embedding. Pooler_output is the embedding of the [CLS] special token. It is considered as a valid representation of the complete sentence.
        cls_x = self.classifier(cls_x)
        # print(cls_x.shape)
        out = self.softmax(cls_x)
        return out


def prediction(model: BertClassifier, out_text: list, tokenizer: BertTokenizer) -> torch.Tensor:
    """
    function for doing prediciton
    :param model: Our model
    :param out_text: toxic text
        :param tokenizer: tokenizer
    :return: probabilities
    """
    with torch.no_grad():
        x = tokenizer(out_text,
                      add_special_tokens=True, max_length=120,
                      truncation=True, padding=True, return_tensors="pt").to(device)
        outputs = model(x["input_ids"], attention_mask=x["attention_mask"])
    return outputs


def inference(model: BertClassifier, input_text: str, tokenizer: BertTokenizer) -> str:
    """
    make inference
    :param model: Our model
    :param input_text: toxic text
    :param tokenizer: tokenizer
    :return: detoxified text
    """
    input_text = clear_str(input_text)
    while True:
        # find most toxic word
        input_text = input_text.split()
        out_text = [" ".join(input_text[:i] + ["<oov>"] + input_text[min(i + 1, len(input_text)):]) for i in range(len(input_text))]
        probs = prediction(model, out_text, tokenizer)
        idx = torch.argmax(probs[:, 0])

        # delete toxic word
        input_text = re.sub("\s*<oov>\s*", " ", out_text[idx]).strip()

        # check if sentence still toxic
        toxicity = prediction(model, [input_text], tokenizer)
        if torch.argmax(toxicity[0]) == 0:
            break

    input_text = re.sub(r'\s([?.!,¿](?:\s|$))', r'\1', input_text)
    return input_text


if __name__ == "__main__":
    toxic_text = input("Write your toxic text: ")
    print("Your toxic text:", toxic_text)

    print("Predicting, please, wait...")

    bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert = BertModel.from_pretrained(bert_model_name)
    model = BertClassifier(bert, 2).to(device)

    model.load_state_dict(torch.load(PATH_OUT + "model-final.pt"))
    model.eval()

    print("Translated text:", inference(model, toxic_text, tokenizer))