Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence

NIST is based on BLEU metric, but with some alternations of weighting averaging on n-gram informativeness.

BLEU, Bilingual Evaluation Understudy score, involves with N-gram of size 1 to 4 overlap between machine translation output and reference 
translation.

Edit Distance measures the minimum number of editing steps to transform output to reference, including the steps of match, 
substitution, insertion and deletion.

Word Error Rate is a similar measurement as the edit distance, except having one extra step of dividing 
the edit distance by length of the reference.

Longest common substring is the number of the longest consecutive string appearing in both reference and the targeted text.

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train  | Dev | Test 
------ |--------| --- | ----
NIST | 0.496  | 0.593 | 0.475
BLEU | 0.612  | 0.433 | 0.603
WER | -0.313 | -0.452| -0.326
LCS | 0.452  | 0.468| 0.500
Edit Dist | -0.002 | -0.175| -0.085

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).


## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.