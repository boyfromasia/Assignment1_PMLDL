import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import os
import warnings
from tqdm import tqdm

nltk.download('stopwords')
stopwords = stopwords.words('english')

warnings.filterwarnings('ignore')

PATH_EXTERNAL = "../../data/external/"
PATH_INHERITIM = "../../data/inheritim/"
PATH_RAW = "../../data/raw/"
PATH_SAVE_FIGURES = "../../reports/figures/"


def preprocess_jigsaw() -> pd.DataFrame:
    """
    open and preprocess jigsaw dataset

    :return: jigsaw dataset
    """
    df = pd.read_csv(os.path.join(PATH_EXTERNAL, 'jigsaw-train.csv'))
    df = df.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
    df['non_toxic'] = 1 - df['toxic']
    df["length"] = df["comment_text"].str.split().apply(len)
    df["avg_word"] = df["comment_text"].str.split().str.len()

    return df


def plot_class_distribution_jigsaw(df: pd.DataFrame):
    """
    plot class distribution of the jigsaw dataset and save it

    :param df: jigsaw dataset
    """
    toxic_num = df[df['toxic'] == 1].shape[0]
    nontoxic_num = df[df['non_toxic'] == 1].shape[0]
    sns.barplot(x=['non_toxic', 'toxic'], y=[nontoxic_num, toxic_num])
    plt.title("Distribution of the classes Jigsaw dataset")
    plt.savefig(PATH_SAVE_FIGURES + "class_distribution_jigsaw.png")
    plt.show()


def plot_word_numbers_jigsaw(df: pd.DataFrame):
    """
    plot histogram of the word numbers using jigsaw dataset and save it

    :param df: jigsaw dataset
    """
    _ = plt.hist(np.log2(df["avg_word"]), color = 'blue', edgecolor = 'black')
    plt.title('Histogram of word numbers in comment text')
    plt.xlabel('$2^{Words}$')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "word_numbers_jigsaw.png")
    plt.show()


def print_info_jigsaw(df: pd.DataFrame):
    """
    print into console information about jigsaw dataset

    :param df: jigsaw dataset
    """
    print()
    print(20 * "=" + "JIGSAW DATASET" + 20 * "=")
    print(df.head(), end="\n\n\n")
    print("Dataset size:", df.shape)
    print("Columns:", df.columns.tolist())

    toxic_num = df[df['toxic'] == 1].shape[0]
    nontoxic_num = df[df['non_toxic'] == 1].shape[0]

    print("Number of nontoxic comments", nontoxic_num / toxic_num, "time greater then number of toxic samples",
          end="\n\n")

    print("Some information about length of JIGSAW dataset.")
    print(df["length"].describe(), end="\n\n\n")

    print("Base information about word number of the sentence.")
    print(df["avg_word"].describe(), end="\n\n\n")


def plot_wordcloud_toxic_jigsaw(df: pd.DataFrame):
    """
    plot wordcloud of toxic words of the jigsaw dataset  and save it

    :param df: jigsaw dataset
    """
    words = []

    for item in df[df["toxic"] == 1].iterrows():
        deleted_punctuation = item[1]["comment_text"].translate(str.maketrans('', '', string.punctuation))
        lowercase = deleted_punctuation.lower()
        tokens = lowercase.split()
        deleted_stopwords = [x for x in tokens if x not in stopwords and not x.isdigit()]
        words.extend(deleted_stopwords)

    wordcloud = WordCloud(collocations=False).generate(" ".join(words))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Wordcloud toxic words Jigsaw dataset")
    plt.savefig(PATH_SAVE_FIGURES + "toxic_wordcloud_jigsaw.png")
    plt.show()


def preprocess_paranmt() -> pd.DataFrame:
    """
    open and preprocess paranmt dataset

    :param df: paranmt dataset
    """
    df = pd.read_table(PATH_RAW + 'filtered.tsv', index_col=0)
    df["avg_word_ref"] = df["reference"].str.split().str.len()
    df["avg_word_trans"] = df["translation"].str.split().str.len()

    return df


def plot_word_numbers_in_reference_paranmt(df: pd.DataFrame):
    """
    plot histogram of the word numbers in 'reference' column and save it

    :param df: paranmt dataset
    """
    _ = plt.hist(np.log2(df["avg_word_ref"]), color='blue', edgecolor='black')
    plt.title('Histogram of word numbers in reference')
    plt.xlabel('$2^{Words}$')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "word_numbers_in_reference_paranmt.png")
    plt.show()


def plot_toxicity_level_reference_text_paranmt(df: pd.DataFrame):
    """
    plot histogram of the toxicity level in 'reference' column and save it

    :param df: paranmt dataset
    """
    _ = plt.hist(df["ref_tox"], color='blue', edgecolor='black')
    plt.title('Histogram of toxicity level of reference text')
    plt.xlabel('level')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "toxicity_level_reference_text_paranmt.png")
    plt.show()


def swap_values_paranmt(df: pd.DataFrame) -> pd.DataFrame:
    """
    swap values in paranmt dataset to make it right,
    since 'reference' column has sentences <= 0.5 toxicity level and
    'translation' column has sentences > 0.5 toxicity level

    :param df:paranmt dataset
    :return: updated paranmt dataset
    """
    df['ref_tox'], df['trn_tox'] = np.where(df.ref_tox > df.trn_tox,
                                            (df['ref_tox'], df['trn_tox']), (df['trn_tox'], df['ref_tox']))

    return df


def plot_final_toxicity_level_reference_text_paranmt(df: pd.DataFrame):
    """
    plot histogram of the toxicity level in 'reference' column of the updated dataset and save it

    :param df: updated paranmt dataset
    """
    _ = plt.hist(df["ref_tox"], color='blue', edgecolor='black')
    plt.title('Histogram of toxicity level of reference text')
    plt.xlabel('level')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "final_toxicity_level_reference_text_paranmt.png")
    plt.show()


def plot_wordcloud_toxic_paranmt(df: pd.DataFrame):
    """
    plot wordcloud of toxic words of the paranmt dataset  and save it

    :param df: paranmt dataset
    """
    words = []

    for item in tqdm(df.iterrows()):
        deleted_punctuation = item[1]["reference"].translate(str.maketrans('', '', string.punctuation))
        lowercase = deleted_punctuation.lower()
        tokens = lowercase.split()
        deleted_stopwords = [x for x in tokens if x not in stopwords and not x.isdigit()]
        words.extend(deleted_stopwords)

    wordcloud = WordCloud().generate(" ".join(words))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Wordcloud toxic words ParaNMT")
    plt.savefig(PATH_SAVE_FIGURES + "toxic_wordcloud_paranmt.png")
    plt.show()


def plot_word_numbers_in_translation_paranmt(df: pd.DataFrame):
    """
    plot histogram of the word numbers in 'translation' column using paranmt dataset and save it

    :param df: paranmt dataset
    """
    _ = plt.hist(np.log2(df["avg_word_trans"]), color='blue', edgecolor='black')
    plt.title('Histogram of word numbers in translation')
    plt.xlabel('$2^{Words}$')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "word_numbers_in_translation_paranmt.png")
    plt.show()


def plot_toxicity_level_translation_text_paranmt(df: pd.DataFrame):
    """
    plot histogram of the toxicity level in 'translation' column using paranmt dataset and save it

    :param df: paranmt dataset
    """
    _ = plt.hist(df["trn_tox"], color='blue', edgecolor='black')
    plt.title('Histogram of toxicity level of translation text')
    plt.xlabel('level')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "toxicity_level_translation_text_paranmt.png")
    plt.show()


def plot_similarity_level_paranmt(df: pd.DataFrame):
    """
    plot histogram of the similarity level between 'translation' and
    'reference' columns using paranmt dataset and save it

    :param df: paranmt dataset
    """
    _ = plt.hist(df["similarity"], color='blue', edgecolor='black')
    plt.title('Histogram of similarity level between reference and translation columns.')
    plt.xlabel('level')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "similarity_level_paranmt.png")
    plt.show()


def plot_length_difference_paranmt(df: pd.DataFrame):
    """
    plot histogram of the length difference between 'translation' and
    'reference' column using paranmt dataset and save it

    :param df: paranmt dataset
    """
    _ = plt.hist(df["lenght_diff"], color='blue', edgecolor='black')
    plt.title('Histogram of length difference number between reference and translation columns.')
    plt.xlabel('level')
    plt.ylabel('samples')
    plt.savefig(PATH_SAVE_FIGURES + "length_difference_paranmt.png")
    plt.show()


def print_info_paranmt(old_df: pd.DataFrame, new_df: pd.DataFrame):
    """
    print into console information about paranmt dataset

    :param old_df: dataset before updating
    :param new_df: dataset after updating
    """
    print()
    print(20 * "=" + "PARANMT DATASET" + 20 * "=")
    print(old_df.head(), end="\n\n\n")
    print("Dataset size:", old_df.shape)
    print("Columns:", old_df.columns.tolist())
    print("Unique reference sentences:", old_df["reference"].unique().shape)

    print("Base information about word number of the sentence in 'reference' column.", end="\n\n")
    print(old_df["avg_word_ref"].describe(), end="\n\n\n")

    print("Base information about toxicity level of reference , etc. 'ref_tox' column.", end="\n\n")
    print(old_df["ref_tox"].describe(), end="\n\n\n")

    para = new_df.reference.value_counts()
    print(para.head(), end="\n\n\n")
    print("Number of unique references with > 1 para:", para[para > 1].shape)
    print("Number of unique references with == 1 para:", para[para == 1].shape)
    print("Avg number of translation per reference with > 1 para:", para[para > 1].mean(), end="\n\n")

    print("Unique translation sentences:", new_df["translation"].unique().shape)
    print(f"Base information about word number of the sentence in 'translation' column.", end="\n\n")
    print(new_df["avg_word_trans"].describe(), end="\n\n\n")

    print("Base information about toxicity level of translation text, etc. 'trn_tox' column.", end="\n\n")
    print(new_df["ref_tox"].describe(), end="\n\n\n")

    print("Base information about similarity level between reference and translation columns.", end="\n\n")
    print(new_df["similarity"].describe(), end="\n\n\n")

    print("Base information about length difference number between reference and translation columns.", end="\n\n")
    print(new_df["lenght_diff"].describe(), end="\n\n\n")


if __name__ == "__main__":
    jigsaw = preprocess_jigsaw()
    plot_class_distribution_jigsaw(jigsaw)
    plot_word_numbers_jigsaw(jigsaw)
    plot_wordcloud_toxic_jigsaw(jigsaw)
    print_info_jigsaw(jigsaw)

    old_paranmt = preprocess_paranmt()
    plot_word_numbers_in_reference_paranmt(old_paranmt)
    plot_toxicity_level_reference_text_paranmt(old_paranmt)

    new_paranmt = swap_values_paranmt(old_paranmt)
    plot_final_toxicity_level_reference_text_paranmt(new_paranmt)
    plot_wordcloud_toxic_paranmt(new_paranmt)
    plot_word_numbers_in_translation_paranmt(new_paranmt)
    plot_toxicity_level_translation_text_paranmt(new_paranmt)
    plot_similarity_level_paranmt(new_paranmt)
    plot_length_difference_paranmt(new_paranmt)

    print_info_paranmt(old_paranmt, new_paranmt)