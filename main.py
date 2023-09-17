from strsimpy.levenshtein import Levenshtein
from strsimpy.metric_lcs import MetricLCS
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.cosine import Cosine
import numpy as np
import pandas as pd
import random
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from suffix_tree import Tree
import os
import csv
import time
from sklearn.utils import shuffle

def get_replace(s):
    for c in ".^$*+?{}[]|()":
        if c in s:
            s = s.replace(c, "\\" + c)
    return s


class ShortPatternMiner:

    def __init__(self, coverage=0.1, coarse_clusterization=['method'], clusterization='kmedoids',
            string_distance='Levenshtein', max_clusters=20):
        # here will be all global parameters
        # used for final pattern generation
        self.coverage = coverage
        self.coarse_clusterization = coarse_clusterization
        self.clusterization = clusterization
        self.string_distance = string_distance
        self.max_clusters = max_clusters

    def _coarse_cluster(self, df):
        param = self.coarse_clusterization
        clusters = []
        if len(param) == 1:
            uniq_val = df[param[0]].unique()
            for val in uniq_val:
                clusters.append(df.loc[(df[param[0]] == val)])
        elif len(param) == 2:
            uniq_val1 = df[param[0]].unique()
            uniq_val2 = df[param[1]].unique()
            for val1 in uniq_val1:
                for val2 in uniq_val2:
                    clusters.append(df.loc[((df[param[0]] == val1) & (df[param[1]] == val2))])

        return clusters

    def _get_distance_for_strings(self, s1, s2):
        distance = -1
        
        if self.string_distance == "Levenshtein":
            levenshtein = Levenshtein()
            distance = levenshtein.distance(s1, s2)
        elif self.string_distance == "MetricLCS":
            metric_lcs = MetricLCS()
            distance = metric_lcs.distance(s1, s2)
        elif self.string_distance == "JaroWinkler":
            jarowinkler = JaroWinkler()
            distance = jarowinkler.distance(s1, s2)
        elif self.string_distance == "Cosine":
            cosine = Cosine(2)
            distance = cosine.distance(s1, s2)

        return distance

    def _get_distance_for_rows(self, row1, row2):
        sum_dist = 0.0
        for i in range(len(row1)):
            sum_dist += self._get_distance_for_strings(str(row1[i]), str(row2[i]))

        return sum_dist

    def _get_distance_matrix(self, df):
        num_of_transactions = len(df)
        matrix = np.zeros((num_of_transactions, num_of_transactions))
        for i in range(num_of_transactions):
            for j in range(i, num_of_transactions):
                r1 = df.iloc[i].to_dict()
                r2 = df.iloc[j].to_dict()
                val1 = list(r1.values())
                val2 = list(r2.values())
                matrix[j][i] = matrix[i][j] = self._get_distance_for_rows(val1, val2)

        return matrix

    def _cluster(self, df):
        clusters_coarse = self._coarse_cluster(df)
        clusters = []
        if self.clusterization == "kmedoids":
            for cluster_coarse in clusters_coarse:
                avg_scores = []
                for num_of_clusters in range(2, self.max_clusters):
                    if len(cluster_coarse) >= num_of_clusters:
                        distance_matrix = self._get_distance_matrix(cluster_coarse)
                        initial_medoids = random.sample(range(len(cluster_coarse)), num_of_clusters)
                        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type="distance_matrix")
                        kmedoids_instance.process()
                        score = silhouette(distance_matrix, clusters, data_type="distance_matrix").process().get_score()
                        avg_scores.append(max(score))
                optimal_k = avg_scores.index(max(avg_scores)) + 2
                if len(cluster_coarse) >= optimal_k:
                    distance_matrix = self._get_distance_matrix(cluster_coarse)
                    initial_medoids = random.sample(range(len(cluster_coarse)), optimal_k)
                    kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type="distance_matrix")
                    kmedoids_instance.process()
                    clust = kmedoids_instance.get_clusters()
                    for cl in clust:
                        clusters.append(cluster_coarse.iloc[cl])
        elif self.clusterization == "hierarchical":
            for cluster_coarse in clusters_coarse:
                avg_scores = []
                for num_of_clusters in range(2, self.max_clusters):
                    if len(cluster_coarse) > num_of_clusters:
                        distance_matrix = self._get_distance_matrix(cluster_coarse)
                        model = AgglomerativeClustering(num_of_clusters, metric="precomputed", linkage='complete')
                        clust = model.fit_predict(distance_matrix)
                        score = davies_bouldin_score(distance_matrix, clust)
                        avg_scores.append(score)
                if avg_scores:                                        # clustering with optimal value of k
                    optimal_k = avg_scores.index(min(avg_scores)) + 2
                    if len(cluster_coarse) > optimal_k:
                        distance_matrix = self._get_distance_matrix(cluster_coarse)
                        model = AgglomerativeClustering(optimal_k, metric="precomputed", linkage='complete')
                        clust = model.fit_predict(distance_matrix)
                        for k in range(optimal_k):
                            cl = [i for i in range(len(clust)) if clust[i] == k]  # get index from one cluster
                            clusters.append(cluster_coarse.iloc[cl])
        return clusters

    def _get_common_substrings(self, column):
        k_min = self.coverage * len(column)
        l_min = 2
        column = column.dropna()
        column_dict = column.to_dict()
        for key, val in column_dict.items():
            column_dict[key] = str(val)
        tree = Tree(column_dict)
        sub_str = []
        common_substr = []
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
                        common_substr.append([path, k1])

        return common_substr

    def _longest_common_suffix(self, list_of_tokens):
        reversed_strings = [s[::-1] for s in list_of_tokens]
        return os.path.commonprefix(reversed_strings)[::-1]

    def _longest_common_prefix(self, list_of_tokens):
        return os.path.commonprefix(list_of_tokens)

    def _merge_tokens_prefix(self, list_of_tokens, common_prefix):
        len_prefix = len(common_prefix)
        ans = common_prefix + '('
        for token in list_of_tokens:
            ans += token[len_prefix:] + '|'
        ans = ans[:len(ans) - 1] + ')'
        return ans

    def _merge_tokens_suffix(self, list_of_tokens, common_suffix):
        len_suffix = len(common_suffix)
        ans = '('
        for token in list_of_tokens:
            ans += token[:len(token) - len_suffix] + '|'
        ans = ans[:len(ans) - 1] + ')' + common_suffix
        return ans    

    def _merging_strings(self, seq):
        sequences = [[i, False] for i in seq]
        seq1 = seq.copy()
        tokens_to_del = []
        tokens_to_add = []
        for i in range(len(sequences)):
            tokens_list_with_common_prefix = []
            tokens_list_with_common_suffix = []
            fl_prefix = False
            fl_suffix = False
            for j in range(i + 1, len(sequences)):
                if not sequences[j][1]:
                    if len(self._longest_common_prefix([sequences[i][0], sequences[j][0]])) >= 2:
                        tokens_list_with_common_prefix.append(sequences[j][0])
                        fl_prefix = True
                        sequences[j][1] = True
                    if len(self._longest_common_suffix([sequences[i][0], sequences[j][0]])) >= 2:
                        fl_suffix = True
                        tokens_list_with_common_suffix.append(sequences[j][0])
                        sequences[j][1] = True
            if fl_suffix:
                tokens_list_with_common_suffix.append(sequences[i][0])
                sequences[i][1] = True
            if fl_prefix:
                tokens_list_with_common_prefix.append(sequences[i][0])
                sequences[i][1] = True

            if 0 < len(tokens_list_with_common_prefix) <= 3:
                tokens_to_del += tokens_list_with_common_prefix
                merged_token = self._merge_tokens_prefix(tokens_list_with_common_prefix,
                        self._longest_common_prefix(tokens_list_with_common_prefix))
                tokens_to_add.append(merged_token)
            if 0 < len(tokens_list_with_common_suffix) <= 3:
                tokens_to_del += tokens_list_with_common_suffix
                merged_token = self._merge_tokens_suffix(tokens_list_with_common_suffix,
                        self._longest_common_suffix(tokens_list_with_common_suffix))
                tokens_to_add.append(merged_token)

        tokens_to_del = list(set(tokens_to_del))

        for item in tokens_to_del:
            seq1.remove(item)
        for item in tokens_to_add:
            seq1.append(item)
        return seq1

    def _get_list_of_tokens_with_offset(self, df, tokens):
        tokens_list = []
        uniq_offsets = []
        tokens_list_changed = []
        for token in tokens:
            offset_list = []
            for d in df:
                if d.find(token) != -1:
                    if not d.find(token) in offset_list:
                        offset_list.append(d.find(token))
                    if not d.find(token) in uniq_offsets:
                        uniq_offsets.append(d.find(token))            
            if len(set(offset_list)) == 0:
                continue
            if len(set(offset_list)) == 1:
                tokens_list.append([token, offset_list, True])
            else:
                tokens_list.append([token, offset_list, False])
        if uniq_offsets:
            for uniq_offset in uniq_offsets:
                tokens_with_uniq_offset = []
                for token, offset_list, position_specific in tokens_list:
                    if position_specific:
                        if uniq_offset in offset_list:
                            tokens_with_uniq_offset.append(token)
                if 1 < len(tokens_with_uniq_offset) < 4:
                    merged_token = '('
                    for token in tokens_with_uniq_offset:
                        merged_token += token + '|'
                        if [token, offset_list, position_specific] in tokens_list:
                            tokens_list.remove([token, offset_list, position_specific])
                    merged_token = merged_token[:len(merged_token) - 1] + ')'
                    if uniq_offset == 0:
                        merged_token = '^' + merged_token
                    tokens_list_changed.append([merged_token, True, True])
        for token, offset_list, position_specific in tokens_list:
            tokens_list_changed.append([token, position_specific, False])
        return tokens_list_changed
    

    def _get_transformation_with_special_characters(self, tokens):
        tokens_changed = []
        dict_with_special_char = {}
        h = ord("А")
        for token in tokens:

            if token[2]:   # special_char
                i = 0
                min_len = float("inf")
                cur_len = 0
                while(token[0][i] != ")"):
                    if token[0][i] == '^' or token[0][i] == '(':
                        i += 1
                    elif token[0][i] == '|':
                        i += 1
                        if cur_len < min_len:
                            min_len = cur_len
                            cur_len = 0
                    else:
                        cur_len += 1
                        i += 1
                
                tokens_changed.append([chr(h) * min_len, token[1], True])
                dict_with_special_char[chr(h)] = token[0]
                h += 1
            else:
                tokens_changed.append(token)
        return tokens_changed, dict_with_special_char

    def _align(x, y, s_match, s_mismatch, s_gap):
        A = []
        for i in range(len(y) + 1):
            A.append([0] * (len(x) + 1))
        for i in range(len(y) + 1):
            A[i][0] = s_gap * i
        for i in range(len(x) + 1):
            A[0][i] = s_gap * i
        for i in range(1, len(y) + 1):
            for j in range(1, len(x) + 1):
                A[i][j] = max(
                    A[i][j - 1] + s_gap,
                    A[i - 1][j] + s_gap,
                    A[i - 1][j - 1] + (s_match if (y[i - 1] == x[j - 1] and y[i - 1] != '-') else 0) + (
                        s_mismatch if (y[i - 1] != x[j - 1] and y[i - 1] != '-' and x[j - 1] != '-') else 0) + (
                        s_gap if (y[i - 1] == '-' or x[j - 1] == '-') else 0)
                )
        align_X = ""
        align_Y = ""
        i = len(x)
        j = len(y)
        while i > 0 or j > 0:
            current_score = A[j][i]
            if i > 0 and j > 0 and (
                    ((x[i - 1] == y[j - 1] and y[j - 1] != '-') and current_score == A[j - 1][i - 1] + s_match) or    # mathc
                    ((y[j - 1] != x[i - 1] and y[j - 1] != '-' and x[i - 1] != '-') and current_score == A[j - 1][    # mismathc
                        i - 1] + s_mismatch) or
                    ((y[j - 1] == '-' or x[i - 1] == '-') and current_score == A[j - 1][i - 1] + s_gap)               # - or symbol
            ):
                align_X = x[i - 1] + align_X
                align_Y = y[j - 1] + align_Y
                i = i - 1
                j = j - 1
            elif i > 0 and (current_score == A[j][i - 1] + s_gap):
                align_X = x[i - 1] + align_X
                align_Y = "-" + align_Y
                i = i - 1
            else:
                align_X = "-" + align_X
                align_Y = y[j - 1] + align_Y
                j = j - 1
        return (align_X, align_Y, A[len(y)][len(x)])


    def _get_center(self, sequences):   # find pair with max score
        score_matrix = {}
        for i in range(len(sequences)):
            score_matrix[i] = {}
        for i in range(len(sequences)):
            for j in range(len(sequences)):
                if i != j:
                    s_match = 1
                    if sequences[i][1] or sequences[j][1]:  # bonus for position
                        s_match += 1
                    score_matrix[i][j] = self._align(sequences[i][0], sequences[j][0], s_match, 0, 0)

        center = 0
        center_score = float('-inf')
        for scores in score_matrix:
            sum = 0
            for i in score_matrix[scores]:
                sum += score_matrix[scores][i][2]
            if sum > center_score:
                center_score = sum
                center = scores
        alignments_needed = {}

        for i in range(len(sequences)):
            if i != center:
                s_match = 1
                if sequences[i][1] or sequences[center][1]:  # bonus for position
                    s_match += 1
                s1, s2, sc = self._align(sequences[i][0], sequences[center][0], s_match, 0, 0)
                alignments_needed[i] = (s2, s1, sc)
        return center, alignments_needed


    def _align_gaps(seq1, seq2, aligneds, new):
        i = 0
        while i < max(len(seq1), len(seq2)):
            try:
                if i > len(seq1) - 1:
                    seq1 = seq1[:i] + "-" + seq1[i:]
                    naligneds = []
                    for seq in aligneds:
                        naligneds.append(seq[:i] + "-" + seq[i:])
                    aligneds = naligneds
                elif i > len(seq2) - 1:
                    seq2 = seq2[:i] + "-" + seq2[i:]
                    new = new[:i] + "-" + new[i:]
                elif (seq1[i] == "-" and i >= len(seq2)) or (seq1[i] == "-" and seq2[i] != "-"):
                    seq2 = seq2[:i] + "-" + seq2[i:]
                    new = new[:i] + "-" + new[i:]
                elif (seq2[i] == "-" and i >= len(seq1)) or (seq2[i] == "-" and seq1[i] != "-"):
                    seq1 = seq1[:i] + "-" + seq1[i:]
                    naligneds = []
                    for seq in aligneds:
                        naligneds.append(seq[:i] + "-" + seq[i:])
                    aligneds = naligneds
            except Exception:
                print("ERROR")
            i += 1

        aligneds.append(new)
        return seq1, aligneds


    def _msa(self, alignments):
        aligned_center = alignments[list(alignments.keys())[0]][0]
        aligneds = []
        aligneds.append(alignments[list(alignments.keys())[0]][1])

        for seq in list(alignments.keys())[1:]:
            cent = alignments[seq][0]
            newseq = alignments[seq][1]
            aligned_center, aligneds = self._align_gaps(aligned_center, cent, aligneds, newseq)

        return aligneds, aligned_center


    def _order_results(aligneds, center_seq, center, sequences):
        i = 0
        j = 0
        results = []
        while i < len(sequences):
            if i == center:
                results.append(center_seq)
                i += 1
            else:
                results.append(aligneds[j])
                i += 1
                j += 1
        return results

    def _get_regexp(self, align_string):
        i = 0
        res = ''
        while i < len(align_string):
            if align_string[i] == 'Ф':
                while align_string[i] == 'Ф':
                    i += 1            
                    if i == len(align_string):
                        res += '.*'
                        break

                if align_string[i] == 'Л':
                    while align_string[i] == 'Л':
                        i += 1
                        if i == len(align_string):
                            break
                    res += '.*'
                else:
                    res += '.*'
            elif align_string[i] == 'Л':
                spec_len = 0
                while align_string[i] == 'Л':
                    i += 1
                    if i == len(align_string):
                        res += ".{" + str(spec_len+1) + "}"
                        break
                    spec_len += 1
                if i == len(align_string):
                    break
                if align_string[i] == 'Ф':
                    while align_string[i] == 'Ф':
                        i += 1
                        if i == len(align_string):
                            break
                    res += '.*'
                else:
                    res += ".{" + str(spec_len) + "}"
            else:
                res += align_string[i]
                i += 1
        return res

    def generate_pattern(self, df):
        # get_header
        file = open("simple_dataset.csv", 'r')
        rows = csv.reader(file)
        headers = next(rows)
        file.close()
        # dict with tokens list for each header
        tokens_to_headers = dict(zip(headers, [[] for i in range(len(headers))]))
        s = 0
        clusters = a._cluster(df)
        for clusterrr in clusters:
            if type(clusterrr) != type([]):
                for name, values in clusterrr.items():
                    common_substr = self._get_common_substrings(values)
                    common_substr = [i for i, j in common_substr]
                    val_arr = values.to_list()
                    val_arr = [str(i) for i in val_arr]
                    for cs in common_substr:
                        if cs not in tokens_to_headers[name]:
                            tokens_to_headers[name].append(cs)
                        
        return tokens_to_headers

# available parameters:
# coarse_clusterization = "method", "protocol", "raw_uri", "x-location", "dst_ip", "src_port", "dst_port"
# string_distance = "Levenshtein", "MetricLCS", "JaroWinkler", "Cosine"
# clusterization = "kmedoids", "hierarchical"


with open("simple_dataset.csv", "r") as data:
    df = pd.read_csv(data)
df1 = shuffle(df)
train_df = df1.iloc[[i for i in range(500)]]
test_df = df1.iloc[[i for i in range(501, 1000)]]

a = ShortPatternMiner(coverage=0.1, coarse_clusterization=["raw_uri"], string_distance="Cosine", clusterization='hierarchical', max_clusters=5)