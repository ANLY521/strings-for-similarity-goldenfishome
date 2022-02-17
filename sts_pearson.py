from scipy.stats import pearsonr
import argparse
from util import parse_sts
from sts_nist import symmetrical_nist
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk.tokenize import word_tokenize


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]
    for score_type in score_types:
        if score_type =="NIST":
            nist_score = [symmetrical_nist(text) for text in texts]
        if score_type == "BLEU":
            bleu_score = [sentence_bleu(text) for text in texts]
        if score_type == "Word Error Rate" or "Edit Distance":
            edit_distance_score = []
            for text in texts:
                source, target = text
                source_toks = word_tokenize(source.lower())
                target_toks = word_tokenize(target.lower())
                n = len(source_toks)
                m = len(target_toks)
                D = np.zeros((n + 1, m + 1))  # create distance matrix

                for i in range(n):
                    D[i, 0] = i
                for j in range(m):
                    D[0, j] = j

                for i in range(n):
                    for j in range(m):
                        if source_toks[i] == target_toks[j]:
                            D[i + 1][j + 1] = D[i][j]
                        else:
                            D[i + 1][j + 1] = min(D[i][j], D[i + 1][j], D[i][j + 1]) + 1
                edit_distance_score.append(D[n,m])

        if score_type == "Longest common substring":
            LCS_score = []
            for text in texts:
                source, target = text
                source_toks = word_tokenize(source.lower())
                target_toks = word_tokenize(target.lower())
                n = len(source_toks)
                m = len(target_toks)
                D = np.zeros((n + 1, m + 1))  # create distance matrix

                for i in range(n):
                    D[i, 0] = i
                for j in range(m):
                    D[0, j] = j

                for i in range(n):
                    for j in range(m):
                        if source_toks[i] == target_toks[j]:
                            D[i + 1][j + 1] = D[i][j]
                        else:
                            D[i + 1][j + 1] = min(D[i][j], D[i + 1][j], D[i][j + 1]) + 1
                LCS_score.append(D.max())


    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        score = 0.0
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

