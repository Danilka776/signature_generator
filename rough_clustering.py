import csv


NUM_OF_TRASACTIONS = 1000
MAX_LEN = 10000 # delete


# Classifies according to the general method.
def Method_classifier(name):
    file = open(name, 'r')
    rows = csv.DictReader(file)
    cluster = dict(POST=[], GET=[])
    for row in rows:
        if row["method"] == 'POST':
            cluster['POST'].append(row)
        elif row["method"] == 'GET':
            cluster['GET'].append(row)
    file.close()
    return cluster


# Classifies according to the general protocol.
def Protocol_classifier(name):
    file = open(name, 'r')
    rows = csv.DictReader(file)
    cluster = {0.9: [], 1.0: [], 1.1: [], 2.0: []}
    for row in rows:
        if row["protocol"] == '0.9':
            cluster[0.9].append(row)
        elif row["protocol"] == '1.0':
            cluster[1.0].append(row)
        elif row["protocol"] == '1.1':
            cluster[1.1].append(row)
        elif row["protocol"] == '2.0':
            cluster[2.0].append(row)
    file.close()
    return cluster


# Classification by similar uri
def Uri_classifer(name):
    file = open(name, 'r')
    rows = csv.DictReader(file)
    cluster = {}
    for row in rows:
        uri = row["raw_uri"].split("/")[1]
        if cluster.get(uri) is None:
            cluster[str(uri)] = [row]
        else:
            cluster[uri].append(row)
    file.close()
    return cluster


# Classification by average length of body
def Len_classifer(name):
    aver_len = Get_average_len(name)
    file = open(name, 'r')
    rows = csv.DictReader(file)
    cluster = {"above": [], "below": []}
    for row in rows:
        if len(row["body"]) < aver_len:
            cluster["below"].append(row)
        else:
            cluster["above"].append(row)
    file.close()
    return cluster


def Get_average_len(name_of_file):
    file = open(name_of_file, 'r')
    rows = csv.DictReader(file)
    sum_len = 0
    min_len = MAX_LEN
    max_len = 0
    for row in rows:
        cur_len = len(row["body"])
        sum_len += cur_len
        if cur_len > max_len:
            max_len = cur_len
        if cur_len < min_len:
            min_len = cur_len
    file.close()
    average_len = sum_len / NUM_OF_TRASACTIONS
    return average_len

