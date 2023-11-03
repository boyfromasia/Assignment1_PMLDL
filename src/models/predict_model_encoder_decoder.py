import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import unicodedata
import re

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PATH_OUT = "../../models/gru-detoxification/"
PATH = "../../data/inheritim/"
BATCH_SIZE = 32
embedding_dim = 256
units = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unicode_to_ascii(s: str) -> str:
    """
    Normalizes latin chars with accent to their canonical decomposition

    :param s: input sentence
    :return: normalized sentence
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w) -> str:
    """
    preprocess the sentence
    :param w: input sentence
    :return: preprocessed sentence
    """
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang: list):
        """ lang are the list of phrases from each language """
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        """ create word2idx, idx2word and vocab """
        for phrase in self.lang:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
        # sort the vocab
        self.vocab = sorted(self.vocab)
        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor: list) -> int:
    """
    calculate the max_length of input and output tensor
    :param tensor: input tensor
    :return: max length
    """
    return max(len(t) for t in tensor)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, batch_sz: int):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)

    def forward(self, x: torch.Tensor, device: torch.device) -> (torch.Tensor, torch.Tensor):
        # x: batch_size, max_length

        # x: batch_size, max_length, embedding_dim
        x = self.embedding(x)

        self.hidden = self.initialize_hidden_state(device)

        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        return output, self.hidden

    def initialize_hidden_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int, enc_units: int, batch_sz: int):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units,
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.enc_units, self.vocab_size)

        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, enc_output: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1,0,2)

        # hidden shape == (batch_size, hidden size) we convert it to (batch_size, 1, hidden size)
        hidden_with_time_axis = hidden.permute(1, 0, 2)

        # score: (batch_size, max_length, hidden_size) # Bahdanaus's
        # It doesn't matter which FC we pick for each of the inputs
        score = self.V(torch.tanh(self.W2(enc_output) + self.W1(hidden_with_time_axis)))

        #attention_weights shape == (batch_size, max_length, 1)
        attention_weights = nn.Softmax(dim=1)(score)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = torch.sum(attention_weights * enc_output, dim=1)

        # pass the context vector into embedding layer
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # concatenate the context vector and x
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        return x, state, attention_weights

    def initialize_hidden_state(self) -> torch.Tensor:
        return torch.zeros((1, self.batch_sz, self.dec_units))


def pad_sequences(x: list, max_len: int) -> list:
    """
    add padding
    :param x: token's sentence
    :param max_len: max length of words from column
    :return: token's sentence with padding
    """
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


def inference(encoder: Encoder, decoder: Decoder, sentence: str, special_tokens: bool=True) -> str:
    """
    get inference of the encoder-decoder model
    :param encoder: our encoder
    :param decoder: our decoder
    :param sentence: toxic sentence
    :param special_tokens: print special tokens or not
    :return: translated text
    """
    encoder.eval()
    decoder.eval()

    sentence = torch.unsqueeze(sentence, dim=1)
    with torch.no_grad():
        enc_output, enc_hidden = encoder(sentence.to(device), device)
        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * 1)

        out_sentence = []
        for t in range(1, sentence.size(0)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                 dec_hidden.to(device),
                                                 enc_output.to(device))
            dec_input = predictions.argmax(dim=1).unsqueeze(1)
            out_word = targ_lang.idx2word[predictions.squeeze().argmax().item()]
            if special_tokens:
                out_sentence.append(out_word)
            else:
                if out_word != "<pad>" and out_word != "<end>":
                    out_sentence.append(out_word)

    out_sentence = " ".join(out_sentence)
    out_sentence = re.sub(r'\s([?.!,¿](?:\s|$))', r'\1', out_sentence)
    return out_sentence


if __name__ == "__main__":
    toxic_text = input("Write your toxic text: ")
    print("Your toxic text:", toxic_text)
    test_sentence = preprocess_sentence(toxic_text)

    print("Predicting, please, wait...")
    df = pd.read_csv(PATH + 'filtered.csv', index_col=0)

    df['reference'] = df['reference'].apply(lambda w: preprocess_sentence(w))
    df['translation'] = df['translation'].apply(lambda w: preprocess_sentence(w))

    # index language using the class above
    inp_lang = LanguageIndex(df['reference'].values.tolist())
    targ_lang = LanguageIndex(df['translation'].values.tolist())

    # Vectorize the input and target languages
    input_tensor = [[inp_lang.word2idx[s] for s in reference.split(' ')]  for reference in df['reference'].values.tolist()]
    target_tensor = [[targ_lang.word2idx[s] for s in translation.split(' ')]  for translation in df['translation'].values.tolist()]

    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE)

    encoder.load_state_dict(torch.load(PATH_OUT + "encoder-final.pt"))
    decoder.load_state_dict(torch.load(PATH_OUT + "decoder-final.pt"))

    encoder.to(device)
    decoder.to(device)

    encoder.batch_sz = 1
    encoder.initialize_hidden_state(device)
    decoder.batch_sz = 1
    decoder.initialize_hidden_state()

    test_sentence = [inp_lang.word2idx[s] for s in test_sentence.split(' ')]
    test_sentence = pad_sequences(test_sentence, max_length_inp)
    ret = inference(encoder, decoder, torch.tensor(test_sentence), special_tokens=False)
    print(ret)
