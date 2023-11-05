# Text De-Toxicfication

### Author

* Nguyen Gia Trong
* BS20-AI
* g.nguyen@innopolis.university

## How use repository?

1. Clone repository: 
```
git clone https://github.com/boyfromasia/Assignment1_PMLDL.git
```

2. Run `main.sh` file to download 
[model checkpoints](https://drive.google.com/file/d/1eMUzG0scjlQjkQB-bP7G_jDPDFDRBDM6/view?usp=drive_link)
and [data](https://drive.google.com/file/d/1BPDecfpB3uaznMTiEP4NZoZCzW0Rc6LE/view?usp=drive_link): 
```
bash main.sh
```

3. Install all the required packages: 

```
pip install -r requirements.txt
```

## Basic usage

### Make dataset

It is not required, since you already downloaded it using `main.sh`,
but if you want you can run it to preprocess data from `./data/raw` and `./data/inheritim` 
to `./data/inheritim`.

```
python src/data/make_dataset.py
```

### Train model
Choose model *(bert, encoder_decoder, t5small)* and replace * with it. 
You can reproduce my experiments.

```
python src/models/train_model_*.py
```

### Predict model 
Choose model *(bert, encoder_decoder, t5small)* and replace * with it.
All instruction will be in console. You can see inference of the models.

```
python src/models/predict_model_*.py
```

### Visualize
You can see EDA part of this project. Also note that the console also displays information.

```
python src/visualization/predict_model_*.py
```
