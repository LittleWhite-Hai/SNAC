# -*- coding: UTF-8 -*-
import time

import networkx as nx
from networkx.algorithms import isomorphism
import PyBliss
import Queue
import numpy as np
import tools
import random
import os

def fine_partition(rough_partition_elem):
    max_edges = len(rough_partition_elem[0]) * (len(rough_partition_elem[0]) - 1) / 2
    if len(rough_partition_elem[0]) <= 3 or max_edges == len(rough_partition_elem[0].edges):
        return [rough_partition_elem]

    result = []
    for j in rough_partition_elem:
        matching_ok = False
        for sub_result in result:
            if isomorphism.faster_could_be_isomorphic(j, sub_result[0]):
                if isomorphism.fast_could_be_isomorphic(j, sub_result[0]):
                    if isomorphism.is_isomorphic(j, sub_result[0]):
                        sub_result.append(j)
                        matching_ok = True
                        break
        if not matching_ok:
            sub_result = []
            sub_result.append(j)
            result.append(sub_result)
    return result


def partition(source):
    G = nx.read_edgelist(source, nodetype=int)
    node_components = nx.connected_components(G)
    graph_components = []
    for i in node_components:
        graph_components.append(G.subgraph(i))
    graph_components = sorted(graph_components, key=lambda gra: (len(gra), len(gra.edges)))
    len_components = len(graph_components)
    rough_partitions = []
    i = 0
    while i < len_components:
        j = i
        sub_partition = []
        while j < len_components - 1 and len(graph_components[j]) == len(graph_components[j + 1]) and len(
                graph_components[j].edges) == len(graph_components[j + 1].edges):
            sub_partition.append(graph_components[j])
            j += 1
        sub_partition.append(graph_components[j])
        rough_partitions.append(sub_partition)
        i = j + 1
    isomorphic_components = []
    for rough_partition_ele in rough_partitions:
        fine_sub_partition = fine_partition(rough_partition_ele)
        for i in fine_sub_partition:
            isomorphic_components.append(i)
    return isomorphic_components


def merge_common_set(mapping_set):
    for value in mapping_set.values():
        common_set = set()
        q = Queue.Queue()
        for i in value:
            common_set.add(i)
            q.put(i)
        while not q.empty():
            next = q.get()
            for i in mapping_set[next]:
                if i not in common_set:
                    common_set.add(i)
                    q.put(i)
        for i in common_set:
            mapping_set[i] = common_set
    return mapping_set


automorphic_mapping = []
symmetric_sum = 0


def report(perm, text=None):
    global automorphic_mapping, symmetric_sum
    record = []
    mapping = automorphic_mapping[-1]
    for key, value in perm.items():
        if key != value:
            if value in mapping[key]:
                print "error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            record.append((key, value))
            symmetric_sum += 1
            mapping[key].add(value)
    automorphic_mapping[-1] = mapping
    # if symmetric_sum % 100 == 0:
    #     print "record: ", record, "  ,sum: ", symmetric_sum,
    # if symmetric_sum % 300 == 0 and symmetric_sum!=0:
    #     print ""


def report_(perm, text=None):
    global automorphic_mapping, symmetric_sum
    for key, value in perm.items():
        if key != value:
            symmetric_sum+=1
            automorphic_mapping[key].add(value)

def get_automorphic_mapping(isomorphic_components):
    global automorphic_mapping, symmetric_sum
    automorphic_mapping = []
    symmetric_sum = 0
    for sub_components in isomorphic_components:
        max_edges = len(sub_components[0]) * (len(sub_components[0]) - 1) / 2
        if len(sub_components[0]) < 3 or max_edges == len(sub_components[0].edges):
            sub_mapping = {}
            for i in sub_components[0]:
                sub_mapping[i] = set()
                for j in sub_components[0]:
                    sub_mapping[i].add(j)
            automorphic_mapping.append(sub_mapping)
        else:
            sub_mapping = {}
            for i in sub_components[0]:
                sub_mapping[i] = set()
                sub_mapping[i].add(i)
            automorphic_mapping.append(sub_mapping)
            bliss_g = PyBliss.Graph()
            nodes = list(sub_components[0].nodes)
            edges = list(sub_components[0].edges)
            for vertex in nodes:
                bliss_g.add_vertex(vertex)
            for edge in edges:
                bliss_g.add_edge(edge[0], edge[1])
            bliss_g.find_automorphisms(report, "Aut gen:")

        automorphic_mapping[-1] = merge_common_set(automorphic_mapping[-1])
    return automorphic_mapping


def get_isomorphic_mapping(isomorphic_components):
    isomorphic_mapping = []
    for sub_components in isomorphic_components:

        sub_mapping = {}
        first_graph = sub_components[0]
        for i in first_graph:
            sub_mapping[i] = set([i])

        for component in sub_components[1:]:
            GM = isomorphism.GraphMatcher(first_graph, component)
            if not GM.is_isomorphic():
                print "BIG ERROR!!!!!!!!"
            GM_mapping = GM.mapping
            for key, value in GM_mapping.items():
                sub_mapping[key].add(value)
        isomorphic_mapping.append(sub_mapping)
    return isomorphic_mapping


def ECM(source_classes, target_classes, true_alignment, strategy):
    # 此函数返回belong_to和target_to,前者是一个关于节点到它等价类集合的映射，
    # 后者是关于节点到对应图中的等价类集合集合的映射，集合中的每个元素由一个tuple组成，tuple[0]是对应的等价类，tuple[1]是对应的weight
    all_classes = []
    visited = set()
    values = source_classes.values()  # + target_classes.values()
    for subset in values:
        if len(visited & subset) == 0:
            all_classes.append(subset)
            visited |= subset

    belong_to = {}
    # belong_to.update(source_classes)
    belong_to.update(target_classes)

    target_to = {}
    for subset in all_classes:
        all_target_classes_with_weight = {}
        for i in subset:
            target_set = belong_to[true_alignment[i]]
            min_of_target_set = min(target_set)
            if min_of_target_set in all_target_classes_with_weight:
                all_target_classes_with_weight[min_of_target_set] += 1
            else:
                all_target_classes_with_weight[min_of_target_set] = 1.0
        target_classes = []
        if strategy == "weight":
            total = len(subset)
            for key in all_target_classes_with_weight.keys():
                target_classes.append((belong_to[key], all_target_classes_with_weight[key] / total))
        else:
            max_weight = 0
            max_key = None
            for key, value in all_target_classes_with_weight.items():
                if value > max_weight:
                    max_weight = value
                    max_key = key
            target_classes.append((belong_to[max_key], 1.0))
        for i in subset:
            target_to[i] = target_classes
    return belong_to, target_to


def from_mapping_to_list(class_mapping):
    visited = set()
    class_list = []
    for k, v in class_mapping.items():
        if k in visited:
            continue
        visited |= v
        class_list.append(list(v))
    return class_list


def generate_equivalence_classes(graph_file_name):
    start = time.time()
    isomorphic_components = partition(graph_file_name)
    isomorphic_mapping = get_isomorphic_mapping(isomorphic_components)
    automorphic_mapping = get_automorphic_mapping(isomorphic_components)
    classes_mapping = tools.merge_mapping(automorphic_mapping, isomorphic_mapping)
    class_list = from_mapping_to_list(classes_mapping)
    finish_time=time.time()-start
    main_component={}
    for i in automorphic_mapping:
        if len(i)>len(main_component):
            main_component=i
    symmetric_sum_in_main_component=0
    for v in main_component.values():
        if len(v)>1: symmetric_sum_in_main_component+=1
    symmetric_sum_total=sum([len(i) for i in class_list if len(i)>1])
    total_node=sum([len(i) for i in class_list])
    print_name=graph_file_name.split("/")[3]+"/"+graph_file_name.split("/")[4]
    print print_name+": total_node: {}, main_node:{}, total_symmetric:{}, main_symmetric:{}, per1:{:.3}, per2:{:.3}" \
        .format(total_node,len(main_component),symmetric_sum_total,symmetric_sum_in_main_component,
                float(len(main_component))/total_node,float(symmetric_sum_in_main_component+1)/(symmetric_sum_total+1))
    print "Time of EVO  : {}, len(class_list): {}".format(finish_time,len(class_list))
    return classes_mapping, class_list


def generate_equivalence_classes_(graph_file_name):
    start=time.time()
    global symmetric_sum,automorphic_mapping
    automorphic_mapping={}
    symmetric_sum=0
    G = nx.read_edgelist(graph_file_name, nodetype=int)
    edges=list(G.edges)
    nodes=list(G.nodes)
    bliss_g = PyBliss.Graph()
    for node in nodes:
        automorphic_mapping[node]= {node}
        bliss_g.add_vertex(node)
    for edge in edges:
        bliss_g.add_edge(edge[0],edge[1])
    bliss_g.find_automorphisms(report_, "Aut gen:")
    automorphic_mapping=merge_common_set(automorphic_mapping)
    class_list = from_mapping_to_list(automorphic_mapping)
    print "Time of Naive: {}, len(class_list): {}".format(time.time()-start,len(class_list))


def calculate_class_num(belong_to):
    class_num = 0
    visited = set()
    symmetric_num = 0
    for k, v in belong_to.items():
        if k in visited:
            continue
        if len(v) > 1:
            symmetric_num += len(v)
        class_num += 1
        visited |= v

    return class_num, symmetric_num


def generate_ECM(folder):
    source_classes, source_class_list = generate_equivalence_classes(folder + "source/following.number")
    target_classes, target_class_list = generate_equivalence_classes(folder + "target/following.number")
    true_alignment = {}
    f = open(folder + "true_alignment.txt", "r")
    for line in f:
        arr = line.strip().split()
        true_alignment[int(arr[0])] = int(arr[1])
        true_alignment[int(arr[1])] = int(arr[0])
    f.close()

    for strategy in ["weight", "voting"]:
        belong_to, target_to = ECM(source_classes, target_classes, true_alignment, strategy)
        np.save(folder + "belong_to", belong_to, allow_pickle=True)
        np.save(folder + "target_to" + "_" + strategy, target_to, allow_pickle=True)
        np.save(folder + "class_list", source_class_list, allow_pickle=True)


def ECAC(result_alignment, belong_to, target_to, way=""):
    score = 0.0
    class_num, symmetric_num = calculate_class_num(belong_to)
    node_num = len(belong_to) / 2
    class_num /= 2

    for source_node, target_node in result_alignment:
        class_of_target_node = belong_to[target_node]
        target_classes_of_source_node = target_to[source_node]
        for target_class in target_classes_of_source_node:
            if class_of_target_node == target_class[0]:
                if way == "+" or way == "++":
                    score += target_class[1] / len(target_class[0])
                else:
                    score += target_class[1]
                break
    if way == "++":
        return score / class_num
    else:
        return score / len(result_alignment)


def generate_ECM_for_another_expe2(dataset):
    header = "../../NetworkAlignment/dataspace/Experiment2/"
    ratios = ["000","005", "010", "015", "020","025"]
    for i in ratios:
        print i,
        source_class_mapping, source_class_list = generate_equivalence_classes \
            (header + dataset + "/SymmeRatio" + i + "/source/edgelist/edgelist")
        target_class_mapping, target_class_list = generate_equivalence_classes \
            (header + dataset + "/SymmeRatio" + i + "/target/edgelist/edgelist")
        true_alignment = {}
        f = open(header + dataset + "/SymmeRatio" + i + "/dictionaries/groundtruth", "r")
        for line in f:
            arr = line.strip('\n').split()
            true_alignment[int(arr[0])] = int(arr[1])
        f.close()
        belong_to, target_to = ECM(source_class_mapping, target_class_mapping, true_alignment, strategy="voting")
        np.save(header + dataset + "/SymmeRatio" + i + "/target_to", target_to, allow_pickle=True)
        np.save(header + dataset + "/SymmeRatio" + i + "/belong_to", belong_to, allow_pickle=True)
        np.save(header + dataset + "/SymmeRatio" + i + "/source/class_list", source_class_list, allow_pickle=True)


def generate_ECM_for_another_expe(dataset):
    header = "../../NetworkAlignment/dataspace/"
    ratios = [ "006", "008"]
    target_class_mapping,target_class_list = generate_equivalence_classes(header + dataset + "/target/edgelist/edgelist")

    np.save(header + dataset + "/target/belong_to", target_class_mapping, allow_pickle=True)
    np.save(header + dataset + "/target/class_list", target_class_list, allow_pickle=True)
    for i in ratios:
        print i,
        source_class_mapping,source_class_list=generate_equivalence_classes(header + dataset + "/noise" + i + "-seed1/edgelist/edgelist")
        true_alignment = {}
        f = open(header + dataset + "/noise" + i + "-seed1/dictionaries/groundtruth", "r")
        for line in f:
            arr = line.strip('\n').split()
            true_alignment[int(arr[0])] = int(arr[1])
        f.close()
        belong_to, target_to = ECM(source_class_mapping, target_class_mapping, true_alignment, strategy="voting")
        np.save(header + dataset + "/noise" + i + "-seed1/target_to", target_to, allow_pickle=True)
        np.save(header + dataset + "/noise" + i + "-seed1/class_list", source_class_list, allow_pickle=True)



if __name__ == "__main__":
    path="./DataSets/Experiment4/"
    dir_list = os.listdir(path)
    for dir in dir_list:
        if dir=='Comparison' or dir=='main.py': continue
        dir_list_sub=os.listdir(path+dir)
        for dir_name in dir_list_sub:
            graph_file = path + dir+"/"+dir_name + "/" + dir_name + ".txt"
            generate_equivalence_classes(graph_file)
            generate_equivalence_classes_(graph_file)
            print "---"*30



