import csv
import random
import copy
import time
import numpy as np


def clusterization(k, matrix_of_similarity):
    cluster = [0 for q in range(k)]  # cluster centers
    cluster_content = [[] for i in range(k)]  # contains transaction numbers
    for q in range(k):
        cluster[q] = random.randint(0, len(matrix_of_similarity) - 1)
    cluster_content = data_distribution(cluster, k, matrix_of_similarity)
    privious_cluster = copy.deepcopy(cluster)
    privious_privious_cluster = copy.deepcopy(privious_cluster)
    stop = 20
    while True:
        cluster = cluster_update(cluster, cluster_content, k, matrix_of_similarity)
        cluster_content = data_distribution(cluster, k, matrix_of_similarity)
        if cluster == privious_cluster or cluster == privious_privious_cluster or stop == 0:
            break
        privious_privious_cluster = copy.deepcopy(privious_cluster)
        privious_cluster = copy.deepcopy(cluster)
        stop -= 1
    return cluster, cluster_content


def cluster_update(cluster, cluster_content, k, matrix_of_similarity):
    for i in range(k):
        updated_parameter = cluster[i]
        closest_centroid = 0
        for j in range(len(cluster_content[i])):
            transaction_number = cluster_content[i][j]
            cur_centroid = 0
            for m in range(len(matrix_of_similarity)):
                cur_centroid += matrix_of_similarity[transaction_number][m]
            if closest_centroid < cur_centroid:
                closest_centroid = cur_centroid
                updated_parameter = transaction_number

        cluster[i] = updated_parameter
    return cluster


def data_distribution(cluster, k, matrix_of_similarity):
    cluster_content = [[] for i in range(k)]

    for i in range(len(matrix_of_similarity)):
        min_distance = float('inf')
        situable_cluster = -1
        for j in range(k):
            distance = matrix_of_similarity[i][cluster[j]]
            if distance < min_distance:
                min_distance = distance
                situable_cluster = j

        cluster_content[situable_cluster].append(i)

    return cluster_content


#   Gap statistic
#   Don't work....
# from gap_statistic import OptimalK
# optimalk = OptimalK(clusterer=clusterization)
# n_clusters = optimalk(np.array(arr_of_transaction), n_refs=3, cluster_array=range(1, 7))
# print(n_clusters)


# Davies Bouldin Index
def dist_in_cluster(i, cluster, cluster_content, matrix_of_similarity):
    aver_dist = 0
    for q in cluster_content[i]:
        aver_dist += matrix_of_similarity[q][cluster[i]]
    return (aver_dist / len(cluster_content[i])) if len(cluster_content[i]) != 0 else 0


def DBI(k, cluster, cluster_content, matrix_of_similarity):
    index = 0
    for i in range(k):
        average_distance_in_i_cluster = dist_in_cluster(i, cluster, cluster_content, matrix_of_similarity)
        max_similarity = -1
        for j in range(k):
            if i != j:
                average_distance_in_j_cluster = dist_in_cluster(j, cluster, cluster_content, matrix_of_similarity)
                cur_similarity = (average_distance_in_i_cluster + average_distance_in_j_cluster) / \
                    matrix_of_similarity[cluster[i]][cluster[j]]
                if max_similarity < cur_similarity:
                    max_similarity = cur_similarity
        index += max_similarity

    return index / k


def Silhouette(k, cluster_content, matrix_of_similarity):
    silhouette_a = [[] for i in range(k)]
    silhouette_b = [[] for i in range(k)]
    silhouette = [[] for i in range(k)]
    for i in range(k):
        for i_obj in cluster_content[i]:         # a(i)
            mean_intra_cluster_distance_i = 0
            for j_obj in cluster_content[i]:
                if i_obj != j_obj:
                    mean_intra_cluster_distance_i += matrix_of_similarity[i_obj][j_obj]
            mean_intra_cluster_distance_i = mean_intra_cluster_distance_i / (len(cluster_content[i]) - 1) \
                if len(cluster_content[i]) > 1 else 0
            silhouette_a[i].append(mean_intra_cluster_distance_i)

        for i_obj in cluster_content[i]:    # b(i)
            mean_nearest_cluster_distance = float('inf')
            for j in cluster_content:
                cur_dist = 0
                change_flag = False
                if j != cluster_content[i]:
                    change_flag = True
                    for j_obj in j:
                        cur_dist += matrix_of_similarity[i_obj][j_obj]
                cur_dist = cur_dist / len(j) if len(j) != 0 else 0
                if cur_dist < mean_nearest_cluster_distance and change_flag:
                    mean_nearest_cluster_distance = cur_dist
            silhouette_b[i].append(mean_nearest_cluster_distance)

        for i_obj in range(len(cluster_content[i])):
            s_i = 0
            if len(cluster_content[i]) > 1:
                s_i = (silhouette_b[i][i_obj] - silhouette_a[i][i_obj]) / \
                    max(silhouette_b[i][i_obj], silhouette_a[i][i_obj])
            silhouette[i].append(s_i)

    return silhouette


# for k in range(2, 20):
#    print("k =", k)
#    a = Silhouette(k)


"""
start = time.time()
end = time.time()
print(end - start)
"""
