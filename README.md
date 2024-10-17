# IR Project Part 1 (Puzzles & Riddles)
For this part of the project we decided to experiment using TF-IDF and BM25

## File: project_part_1.py
### To Run
Execute the Python file in the same location as topics_1.json and Answers.json


### Output (Terminal output and two results files):
Terminal: Results based on the two models 

BM25_results.txt: Search results for the BM25 model

TFIDF_results.txt: Search results for the TFIDF model

## Requirements
### Libraries needed:
pyterrier

pandas

nltk

bs4

### Other requirements:
Topics files in the form:

        [
            {
                "Id": topic id string,
                "Title": topic title string,
                "Body": topic body string,
                "Tags": topic tags list
            }
            ...
        ]
Answer file to be in the form:

        [
            {
                "Id": answer id string,
                "Text": answer test string
                "Score": answer score string
            }
            ...
        ]
