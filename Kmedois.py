# from nltk.metrics import edit_distance
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
import csv


np.set_printoptions(precision=7, floatmode='fixed', suppress=True)


def Kmedoids(num_of_clusters, arr_of_transaction, matrix_of_similarity):
    print(f"K = {num_of_clusters}")
    initial_medoids = random.sample(range(len(arr_of_transaction)), num_of_clusters)
    kmedoids_instance = kmedoids(matrix_of_similarity, initial_medoids, data_type="distance_matrix")
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()
    print("Clusters:\n", [[i for i in c] for c in clusters])

    medoids = kmedoids_instance.get_medoids()
    print("Final medoids:\n", [(i) for i in medoids])

    score = silhouette(matrix_of_similarity, clusters, data_type="distance_matrix").process().get_score()
    print(score)
    print(f"max score - {max(score)}")
