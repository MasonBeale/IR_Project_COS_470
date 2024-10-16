import pyterrier as pt
import pandas as pd
import json
if not pt.started():
  pt.init()
# Reading the answers from the json file and converting them to a list of dictionaries
answers_list = json.load(open('Answers.json'))
for answer_document in answers_list:
  answer_document['docno'] = answer_document.pop('Id')
  answer_document['body'] = answer_document.pop('Text')
  answer_document.pop('Score')
answers_list = answers_list[:1000]
# Indexing the answer documents
indexer = pt.IterDictIndexer('./answers_index')
index_ref = indexer.index(answers_list)

# Loading the query file
topics_1_list = json.load(open('topics_1.json'))
topics_2_list = json.load(open('topics_2.json'))

# Creating a list of dictionaries with the query id and the query title for topic 1
queries_title = []
for query in topics_1_list:
  queries_title.append({
    'qid': query['Id'],
    'query': query['Title']
  })
# Creating a list of dictionaries with the query id and the query body for topic 1
queries_body = []
for query in topics_1_list:
  queries_body.append({
    'qid': query['Id'],
    'query': query['Body']
  })
# Creating a list of dictionaries with the query id and the query title and body for topic 1
queries_title_body = []
for query in topics_1_list:
  queries_title_body.append({
    'qid': query['Id'],
    'query': f"{query['Title']} {query['Body']}"
  })

# Creating list of dictionaries with the query id and the query title for topic 2
queries_title_2 = []
for query in topics_2_list:
  queries_title_2.append({
    'qid': query['Id'],
    'query': query['Title']
  })
# Creating list of dictionaries with the query id and the query body for topic 2
queries_body_2 = []
for query in topics_2_list:
  queries_body_2.append({
    'qid': query['Id'],
    'query': query['Body']
  })
# Creating list of dictionaries with the query id and the query title and body for topic 2
queries_title_body_2 = []
for query in topics_2_list:
  queries_title_body_2.append({
    'qid': query['Id'],
    'query': f"{query['Title']} {query['Body']}"
  })

#BM25 Retrieval
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
# Topic 1 results
bm25_query_results_title = bm25.transform(pd.DataFrame(queries_title))
bm25_query_results_title.to_tsv('bm25_query_results_title.tsv', sep='\t', index=False)

bm25_query_results_body = bm25.transform(pd.DataFrame(queries_body))
bm25_query_results_body.to_tsv('bm25_query_results_body.tsv', sep='\t', index=False)

bm25_query_results_title_body = bm25.transform(pd.DataFrame(queries_title_body))
bm25_query_results_title_body.to_tsv('bm25_query_results_title_body.tsv', sep='\t', index=False)

# Topic 2 results
bm25_query_results_title_2 = bm25.transform(pd.DataFrame(queries_title_2))
bm25_query_results_title_2.to_tsv('bm25_query_results_title_2.tsv', sep='\t', index=False)

bm25_query_results_body_2 = bm25.transform(pd.DataFrame(queries_body_2))
bm25_query_results_body_2.to_tsv('bm25_query_results_body_2.tsv', sep='\t', index=False)

bm25_query_results_title_body_2 = bm25.transform(pd.DataFrame(queries_title_body_2))
bm25_query_results_title_body_2.to_tsv('bm25_query_results_title_body_2.tsv', sep='\t', index=False)

#TF-IDF Retrieval
tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
# Topic 1 results
tfidf_query_results_title = tfidf.transform(pd.DataFrame(queries_title))
tfidf_query_results_title.to_tsv('tfidf_query_results_title.tsv', sep='\t', index=False)

tfidf_query_results_body = tfidf.transform(pd.DataFrame(queries_body))
tfidf_query_results_body.to_tsv('tfidf_query_results_body.tsv', sep='\t', index=False)

tfidf_query_results_title_body = tfidf.transform(pd.DataFrame(queries_title_body))
tfidf_query_results_title_body.to_tsv('tfidf_query_results_title_body.tsv', sep='\t', index=False)
# Topic 2 results
tfidf_query_results_title_2 = tfidf.transform(pd.DataFrame(queries_title_2))
tfidf_query_results_title_2.to_tsv('tfidf_query_results_title_2.tsv', sep='\t', index=False)

tfidf_query_results_body_2 = tfidf.transform(pd.DataFrame(queries_body_2))
tfidf_query_results_body_2.to_tsv('tfidf_query_results_body_2.tsv', sep='\t', index=False)

tfidf_query_results_title_body_2 = tfidf.transform(pd.DataFrame(queries_title_body_2))
tfidf_query_results_title_body_2.to_tsv('tfidf_query_results_title_body_2.tsv', sep='\t', index=False)


# Load the QREL file
qrel_df = pd.read_csv('qrel_1.tsv', sep='\t', header=None, names=['query_id', 'ignore', 'document_id', 'relevance_level'])

# Drop the 'ignore' column as it's not needed
qrel_df.drop(columns=['ignore'], inplace=True)

# Load the retrieval results from TSV BM25 - Topic 1 - might not even be necessary to grab from tsv 
bm25_query_results_title_df = pd.read_csv('bm25_query_results_title.tsv', sep='\t')
bm25_query_results_title_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
bm25_query_results_title_comparison_df = pd.merge(bm25_query_results_title_df, qrel_df, on=['query_id', 'document_id'], how='left')

bm25_query_results_body_df = pd.read_csv('bm25_query_results_body.tsv', sep='\t')
bm25_query_results_body_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
bm25_query_results_body_comparison_df = pd.merge(bm25_query_results_body_df, qrel_df, on=['query_id', 'document_id'], how='left')

bm25_query_results_title_body_df = pd.read_csv('bm25_query_results_title_body.tsv', sep='\t')
bm25_query_results_title_body_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
bm25_query_results_title_body_comparison_df = pd.merge(bm25_query_results_title_body_df, qrel_df, on=['query_id', 'document_id'], how='left')

# Load the retrieval results from TSV BM25 - Topic 2 - might not even be necessary to grab from tsv
bm25_query_results_title_2_df = pd.read_csv('bm25_query_results_title_2.tsv', sep='\t')
bm25_query_results_title_2_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
bm25_query_results_title_2_comparison_df = pd.merge(bm25_query_results_title_2_df, qrel_df, on=['query_id', 'document_id'], how='left')

bm25_query_results_body_2_df = pd.read_csv('bm25_query_results_body_2.tsv', sep='\t')
bm25_query_results_body_2_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
bm25_query_results_body_2_comparison_df = pd.merge(bm25_query_results_body_2_df, qrel_df, on=['query_id', 'document_id'], how='left')

bm25_query_results_title_body_2_df = pd.read_csv('bm25_query_results_title_body_2.tsv', sep='\t')
bm25_query_results_title_body_2_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
bm25_query_results_title_body_2_comparison_df = pd.merge(bm25_query_results_title_body_2_df, qrel_df, on=['query_id', 'document_id'], how='left')

# Load the retrieval results from TSV TF-IDF - Topic 1 - might not even be necessary to grab from tsv
tfidf_query_results_title_df = pd.read_csv('tfidf_query_results_title.tsv', sep='\t')
tfidf_query_results_title_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
tfidf_query_results_title_comparison_df = pd.merge(tfidf_query_results_title_df, qrel_df, on=['query_id', 'document_id'], how='left')

tfidf_query_results_body_df = pd.read_csv('tfidf_query_results_body.tsv', sep='\t')
tfidf_query_results_body_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
tfidf_query_results_body_comparison_df = pd.merge(tfidf_query_results_body_df, qrel_df, on=['query_id', 'document_id'], how='left')

tfidf_query_results_title_body_df = pd.read_csv('tfidf_query_results_title_body.tsv', sep='\t')
tfidf_query_results_title_body_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
tfidf_query_results_title_body_comparison_df = pd.merge(tfidf_query_results_title_body_df, qrel_df, on=['query_id', 'document_id'], how='left')

# Load the retrieval results from TSV TF-IDF - Topic 2 - might not even be necessary to grab from tsv
tfidf_query_results_title_2_df = pd.read_csv('tfidf_query_results_title_2.tsv', sep='\t')
tfidf_query_results_title_2_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
tfidf_query_results_title_2_comparison_df = pd.merge(tfidf_query_results_title_2_df, qrel_df, on=['query_id', 'document_id'], how='left')

tfidf_query_results_body_2_df = pd.read_csv('tfidf_query_results_body_2.tsv', sep='\t')
tfidf_query_results_body_2_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
tfidf_query_results_body_2_comparison_df = pd.merge(tfidf_query_results_body_2_df, qrel_df, on=['query_id', 'document_id'], how='left')

tfidf_query_results_title_body_2_df = pd.read_csv('tfidf_query_results_title_body_2.tsv', sep='\t')
tfidf_query_results_title_body_2_df.rename(columns={'qid': 'query_id', 'docno': 'document_id'}, inplace=True)
tfidf_query_results_title_body_2_comparison_df = pd.merge(tfidf_query_results_title_body_2_df, qrel_df, on=['query_id', 'document_id'], how='left')

# Evaluation Metrics
metrics = {
    'nDCG@5': pt.metrics.NDCG(k=5),
    'nDCG@10': pt.metrics.NDCG(k=10),
    'p@5': pt.metrics.Precision(k=5),
    'p@10': pt.metrics.Precision(k=10),
    'MAP': pt.metrics.MeanAveragePrecision(),
    'bpref': pt.metrics.BPref()
}

def evaluate_model(results_df, qrel_df):
    # Replace NaN relevance_level with 0 (not relevant) for easier calculations
    results_df['relevance_level'].fillna(0, inplace=True)

    # Create a binary column indicating whether the document is relevant
    results_df['is_relevant'] = results_df['relevance_level'].apply(lambda x: 1 if x > 0 else 0)

    # Calculate metrics
    evaluation_results = {}
    
    for metric_name, metric in metrics.items():
        evaluation_results[metric_name] = metric.evaluate(results_df[['query_id', 'document_id', 'is_relevant']])
    
    return evaluation_results

# Evaluate BM25 Results Topic 1
bm25_results = {
    'title': evaluate_model(bm25_query_results_title_comparison_df, qrel_df),
    'body': evaluate_model(bm25_query_results_body_comparison_df, qrel_df),
    'title_body': evaluate_model(bm25_query_results_title_body_comparison_df, qrel_df)
}

# Evaluate BM25 Results Topic 2
bm25_results_2 = {
    'title': evaluate_model(bm25_query_results_title_2_comparison_df, qrel_df),
    'body': evaluate_model(bm25_query_results_body_2_comparison_df, qrel_df),
    'title_body': evaluate_model(bm25_query_results_title_body_2_comparison_df, qrel_df)
}

# Evaluate TF-IDF Results Topic 1
tfidf_results = {
    'title': evaluate_model(tfidf_query_results_title_comparison_df, qrel_df),
    'body': evaluate_model(tfidf_query_results_body_comparison_df, qrel_df),
    'title_body': evaluate_model(tfidf_query_results_title_body_comparison_df, qrel_df)
}

# Evaluate TF-IDF Results Topic 2
tfidf_results_2 = {
    'title': evaluate_model(tfidf_query_results_title_2_comparison_df, qrel_df),
    'body': evaluate_model(tfidf_query_results_body_2_comparison_df, qrel_df),
    'title_body': evaluate_model(tfidf_query_results_title_body_2_comparison_df, qrel_df)
}

# Print Evaluation Results - Topic 1
print("BM25 Evaluation Results Topic 1:")
for k, v in bm25_results.items():
    print(f"{k}: {v}")

print("\nTF-IDF Evaluation Results Topic 1:")
for k, v in tfidf_results.items():
    print(f"{k}: {v}")

# Print Evaluation Results - Topic 2
print("\nBM25 Evaluation Results Topic 2:")
for k, v in bm25_results_2.items():
    print(f"{k}: {v}")

print("\nTF-IDF Evaluation Results Topic 2:")
for k, v in tfidf_results_2.items():
    print(f"{k}: {v}")


