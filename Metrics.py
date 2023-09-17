import Levenshtein
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def similarity(transactions):
    num_of_transactions = len(transactions)
    matrix = np.zeros((num_of_transactions, num_of_transactions))
    for i in range(num_of_transactions):
        for j in range(i, num_of_transactions):
            normalized1 = transactions[i].lower()
            normalized2 = transactions[j].lower()
            matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
            matrix[j][i] = matrix[i][j] = matcher.ratio()
    return matrix


def get_similarity(string1, string2):
    str1 = string1.replace(" ", "")
    str2 = string2.replace(" ", "")
    sequens = 0
    x = max(len(str1), len(str2))
    for char1 in str1:
        for i, char2 in enumerate(str2):
            if char1 == char2:
                str2 = str2[:i] + str2[i + 1:]
                sequens += 1
                break
    similarity = sequens / x
    return similarity


def improved_metric(transactions):
    num_of_transactions = len(transactions)
    matrix = np.zeros((num_of_transactions, num_of_transactions))
    for i in range(num_of_transactions):
        for j in range(i, num_of_transactions):
            matrix[j][i] = matrix[i][j] = get_similarity(transactions[i], transactions[j])
    return matrix


def cos_similarity(transactions):
    num_of_transactions = len(transactions)
    vectorizer = CountVectorizer().fit_transform(transactions)
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return (np.ones((num_of_transactions, num_of_transactions)) - cos_sim)


def levenshtein(transactions):
    num_of_transactions = len(transactions)
    matrix = np.zeros((num_of_transactions, num_of_transactions))
    for i in range(num_of_transactions):
        for j in range(i, num_of_transactions):
            matrix[j][i] = Levenshtein.distance(transactions[i], transactions[j]) / \
                len(transactions[i] + transactions[j])
            matrix[i][j] = matrix[j][i]
    return matrix


# print(csim)
# print(cosine_sim_vectors(vectors[1], vectors[999]))
# print(1 - Levenshtein.distance(arr_of_transaction[0], arr_of_transaction[1]) /
#   len(arr_of_transaction[0]+arr_of_transaction[1]))
# get_similarity(arr_of_transaction[0], arr_of_transaction[1])
# print(similarity(arr_of_transaction[0], arr_of_transaction[1]))
# print(levenshtein(arr_of_transaction))
