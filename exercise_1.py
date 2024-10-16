import pyterrier as pt
import pandas as pd
import json
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # explicitly import stopwords
stop_words = stopwords.words('english')
nltk.download('punkt')
from nltk.tokenize import word_tokenize # explicitly import word_tokenize
import bs4
from bs4 import BeautifulSoup
if not pt.java.started():
  pt.java.init()

# Reading the answers from the json file and converting them to a list of dictionaries
with open('Answers.json', 'r', encoding='utf-8') as f1:
    answers_list = json.load(f1)
with open('topics_1.json', 'r', encoding='utf-8') as f2:
    topics_1_list = json.load(f2)
with open('topics_2.json', 'r', encoding='utf-8') as f3:
    topics_2_list = json.load(f3)

# for testing REMOVE LATER
# answers_list = answers_list[:1000]

# fix answers syntax
answers_list = [{"docno" : answer["Id"], "text" : answer["Text"]} for answer in answers_list]

# CHANGE THIS AROUND TO DO DIFFERENT TESTS
topics_1_list = [{"qid" : topic["Id"], "query" : topic["Body"]} for topic in topics_1_list]
topics_2_list = [{"qid" : topic["Id"], "query" : topic["Body"]} for topic in topics_2_list]
# print(answers_list[0])


indexer = pt.IterDictIndexer("./index", overwrite=True)
index_ref = indexer.index(answers_list)
# print(index_ref.toString())

index = pt.IndexFactory.of(index_ref)
BM25_retriever = pt.terrier.Retriever(index, wmodel="BM25")
TFIDF_retriever = pt.terrier.Retriever(index, wmodel="TF_IDF")
# BM25_retriever.search("sudoku")
# TFIDF_retriever.search("sudoku")

# remove html tags from a string of texts returns another string
def remove_tags(soup):
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

# take a list of words, and a list of stop words, remove any stopwords from word list
# returns string with no stop words
def remove_stopwords(word_list):
    post_removal = [word for word in word_list if not word.lower() in stop_words]
    return ' '.join(post_removal).strip()

lst_queries = []
for item in topics_1_list:
  lst_queries.append([item['qid'], remove_stopwords(remove_tags(BeautifulSoup(item['query'].lower(), "html.parser")).split()).translate(str.maketrans('', '', string.punctuation))])
queries = pd.DataFrame(lst_queries, columns=["qid", "query"])
# print(queries)
# BM25_retriever(queries)
# TFIDF_retriever(queries)

# remove html tags from a string of texts returns another string
def remove_tags(soup):
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

# take a list of words, and a list of stop words, remove any stopwords from word list
# returns string with no stop words
def remove_stopwords(word_list):
    post_removal = [word for word in word_list if not word.lower() in stop_words]
    return ' '.join(post_removal).strip()

lst_queries = []
for item in topics_1_list:
  lst_queries.append([item['qid'], remove_stopwords(remove_tags(BeautifulSoup(item['query'].lower(), "html.parser")).split()).translate(str.maketrans('', '', string.punctuation))])
queries = pd.DataFrame(lst_queries, columns=["qid", "query"])
# print(queries)
# BM25_retriever(queries)
# TFIDF_retriever(queries)

with open('qrel_1.tsv', 'r', encoding='utf-8') as f4:
    qrels = pd.read_csv(f4, sep="\t", header=None, names=["qid", "iter", "docno", "label"])
qrels["qid"] = qrels["qid"].astype(str)
qrels["docno"] = qrels["docno"].astype(str)
results = pt.Experiment(
    [BM25_retriever, TFIDF_retriever],
    queries,
    qrels,
    eval_metrics=[
        "ndcg_cut_5", "ndcg_cut_10", "ndcg",   # nDCG@5, nDCG@10, nDCG@all
        "P_5", "P_10", "P",                    # Precision@5, @10, @all
        "map",                                 # Mean Average Precision
        "bpref"                                # Binary Preference
    ],
    names=['BM25', 'TFIDF']
)
print(results)