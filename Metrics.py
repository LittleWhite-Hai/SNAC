import ECAC
import numpy as np
import tools


def test_ECAC(folder, result_alignment_file="result_alignment.txt", strategy="weight", way=""):
    belong_to_file = folder + "belong_to.npy"
    target_to_file = folder + "target_to_" + strategy + ".npy"
    f = open(folder + result_alignment_file, 'r')
    result_alignment = []
    for line in f.readlines():
        line = line.strip()
        k = int(line.split('--')[0])
        v = int(line.split('--')[1])
        result_alignment.append((k, v))
    f.close()
    belong_to = np.load(belong_to_file, allow_pickle=True).item()
    target_to = np.load(target_to_file, allow_pickle=True).item()
    ecac = ECAC.ECAC(result_alignment, belong_to, target_to, way)
    # print("The ECAC is: " + str(ac))
    return ecac


def test_AC(folder="Models/IONE/IONEData/Arenas/Noise000/result/", result_alignment_file="source-target-1.mapping"):
    result_alignment = tools.alignment_file_to_list(folder + result_alignment_file)
    true_alignment = tools.alignment_file_to_map(folder + "true_alignment.txt")
    score = 0
    for alignment in result_alignment:
        if alignment[1] == true_alignment[alignment[0]]:
            score += 1
    # print "The AC is: "+str(float(score) / len(result_alignment))
    return float(score) / len(result_alignment)


