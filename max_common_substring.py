import csv
from itertools import groupby
# from difflib import SequenceMatcher
from suffix_tree import Tree


def get_replace(s):
    for c in "\.^$*+?{}[]|()":
        if c in s:
            s = s.replace(c, "\\" + c)
    return s




def Get_tokens_from_cluster(cluster_id, headers, arr_of_transaction):
    arr = [arr_of_transaction[i] for i in cluster_id]
    values = [[] for _ in range(len(headers))]
    for i in range(len(headers)):  # make list of values for each header
        for transaction in arr:
            values[i].append(transaction[i])

    k = len(arr)
    k_min = int(k / 10)
    l_min = 2

    # dict with header and transaction which got into the cluster
    transaction_in_cluster = dict(zip(headers, values))

    # dict with tokens list for each header
    tokens_to_headers = dict(zip(headers, [[] for i in range(len(headers))]))
    for header in headers:
        tree = Tree()
        for i, transaction in enumerate(transaction_in_cluster[header]):
            tree.add(i, transaction)
        sub_str = []
        # k1 - the number of tokens in which path1 occurs
        for k1, length, path1 in sorted(tree.common_substrings(), reverse=True):
            if k1 < k_min:  # coverage requirements
                break
            if path1 is not None:
                path = str(path1).replace(" ", "")
                fl = True
                for p in sub_str:
                    if path == p:
                        fl = False
                if fl:
                    sub_str.append(path)
                    if path and len(path) >= l_min:
                        tokens_to_headers[header].append(path)
    return tokens_to_headers


"""
tokens_in_transaction = dict()
transaction_to_tokens = dict()  # shows which transactions have this token
for p, _ in S:
    transaction_to_tokens[p] = []

for i, transaction in enumerate(arr):
    tokens_in_transaction[i] = []
    for path, _ in S:
        if transaction[0].find(path) != -1:
            # first - token, second - position where the token occurs
            tokens_in_transaction[i].append([path, transaction.find(path)])

            transaction_to_tokens[path].append([i, transaction.find(path)])

"""


"""
string1 = arr_of_transaction[0]
string2 = arr_of_transaction[1]
match = SequenceMatcher(None, string1, string2).find_longest_match()

print(match)
print(string1[match.a:match.a + match.size])
print(string2[match.b:match.b + match.size])
"""


"""
######
from suffix_trees import STree

a = [arr_of_transaction[i] for i in range(50)]
st = STree.STree(a)
print(len(st.lcs()))
print(st.lcs())
"""
