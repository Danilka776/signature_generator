import os


def global_align(x, y, s_match, s_mismatch, s_gap):
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
                ((x[i - 1] == y[j - 1] and y[j - 1] != '-') and current_score == A[j - 1][i - 1] + s_match) or    # match
                ((y[j - 1] != x[i - 1] and y[j - 1] != '-' and x[i - 1] != '-') and current_score == A[j - 1][    # mismatch
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


def get_center(sequences):   # find pair with max score
    score_matrix = {}
    for i in range(len(sequences)):
        score_matrix[i] = {}
    for i in range(len(sequences)):
        for j in range(len(sequences)):
            if i != j:
                s_match = 1
                if sequences[i][1] or sequences[j][1]:  # bonus for position
                    s_match += 1
                score_matrix[i][j] = global_align(sequences[i][0], sequences[j][0], s_match, 0, 0)

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
            s1, s2, sc = global_align(sequences[i][0], sequences[center][0], s_match, 0, 0)
            alignments_needed[i] = (s2, s1, sc)
    return center, alignments_needed


def align_gaps(seq1, seq2, aligneds, new):
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


def msa(alignments):
    aligned_center = alignments[list(alignments.keys())[0]][0]
    aligneds = []
    aligneds.append(alignments[list(alignments.keys())[0]][1])

    for seq in list(alignments.keys())[1:]:
        cent = alignments[seq][0]
        newseq = alignments[seq][1]
        aligned_center, aligneds = align_gaps(aligned_center, cent, aligneds, newseq)

    return aligneds, aligned_center


def order_results(aligneds, center_seq, center, sequences):
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


def longest_common_suffix(list_of_tokens):
    reversed_strings = [s[::-1] for s in list_of_tokens]
    return os.path.commonprefix(reversed_strings)[::-1]


def longest_common_prefix(list_of_tokens):
    return os.path.commonprefix(list_of_tokens)


def merge_tokens_prefix(list_of_tokens, common_prefix):
    len_prefix = len(common_prefix)
    ans = common_prefix + '('
    for token in list_of_tokens:
        ans += token[len_prefix:] + '|'
    ans = ans[:len(ans) - 1] + ')'
    return ans


def merge_tokens_suffix(list_of_tokens, common_suffix):
    len_suffix = len(common_suffix)
    ans = '('
    for token in list_of_tokens:
        ans += token[:len(token) - len_suffix] + '|'
    ans = ans[:len(ans) - 1] + ')' + common_suffix
    return ans


if __name__ == '__main__':
    #seq = ['TGTTACGG', "GGTTGACTA"]  # TGTT-ACGG ~ GGTTGACTA  --- TRUE
    #seq = ['ACACACTA', "AGCACACA"]  # A-CACACTA ~ AGCACAC-A  --- TRUE
    #seq = [['АААААААААААААААААААА', True, True], ['0000', False, False], ['00000', False, False], ['000000', True, False]]
    seq = [['ACACACTA', True, True], ['AGCACACA', False, False]]
    center, alignments = get_center(seq)
    aligneds, center_seq = msa(alignments)
    results = order_results(aligneds, center_seq, center, seq)
    for i in results:
        print(i)
    results = sorted(results)
    for i in results:
        print(i)
    align_string = results[0]
    cur_str = results[0]
    for i in range(1, len(results)):
        for j in range(len(cur_str)):
            if cur_str[j] == results[i][j]:
                align_string = align_string[:j] + cur_str[j] + align_string[j+1:]
            elif cur_str[j] == '-' or results[i][j] == '-' or cur_str[j] == 'Ф' or results[i][j] == 'Ф':
                align_string = align_string[:j] + 'Ф' + align_string[j+1:]
            else:
                align_string = align_string[:j] + 'Л' + align_string[j+1:]
        print(align_string, i)
    print("-----------")





# print(len(sequences))
# print(len(results))
