import zipfile
import pandas as pd
import numpy as np
import os

PATH_EXTERNAL = "../../data/external/"
PATH_RAW = "../../data/raw/"
PATH_OUTPUT = "../../data/inheritim/"


def unzip_paranmt():
    path_immutable_file = PATH_RAW + "filtered_paranmt.zip"
    path_unzipped_file = PATH_RAW

    with zipfile.ZipFile(path_immutable_file, 'r') as zip_ref:
        zip_ref.extractall(path_unzipped_file)


def make_dataset_jigsaw():
    train_df = pd.read_csv(os.path.join(PATH_EXTERNAL, 'jigsaw-train.csv'))
    train_df = train_df.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
    train_df['non_toxic'] = 1 - train_df['toxic']

    train_df.to_csv(PATH_OUTPUT + "jigsaw.csv")


def make_dataset_paranmt():
    df = pd.read_table(PATH_RAW + 'filtered.tsv', index_col=0)
    df['ref_tox'], df['trn_tox'] = np.where(df.ref_tox > df.trn_tox, (df['ref_tox'], df['trn_tox']),
                                            (df['trn_tox'], df['ref_tox']))

    df.to_csv(PATH_OUTPUT + "filtered.csv")


if __name__ == "__main__":
    unzip_paranmt()
    make_dataset_jigsaw()
    make_dataset_paranmt()