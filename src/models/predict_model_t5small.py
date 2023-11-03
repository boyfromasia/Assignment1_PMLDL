from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch import cuda

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if cuda.is_available() else 'cpu'
SAVE_PATH = "../../models/t5-detoxification/checkpoint-final/"
model_checkpoint = "t5-small"


def inference(model: AutoModelForSeq2SeqLM, inference_request: str, tokenizer: AutoTokenizer) -> str:
    """
    get inference of the t5 model
    :param model: fine-tuned model
    :param inference_request: toxic text
    :param tokenizer: t5 tokenizer
    :return: translated text
    """
    inputs = tokenizer(inference_request, return_tensors="pt").to(device)
    inputs = {k: v for k, v in inputs.items()}

    outputs = model.generate(**inputs, num_beams=1, do_sample=False)
    for ex in outputs:
        return tokenizer.decode(ex, skip_special_tokens=True)


if __name__ == "__main__":
    toxic_text = input("Write your toxic text: ")
    print("Your toxic text:", toxic_text)

    print("Predicting, please, wait...")

    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_PATH).to(device)
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("Translated text:", inference(model, toxic_text, tokenizer))
