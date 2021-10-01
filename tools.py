# -*- coding: UTF-8 -*-
import random
import networkx as nx


def alignment_file_to_map(true_alignment_file):
    f = open(true_alignment_file, "r")
    true_alignment = {}
    for line in f.readlines():
        line = line.strip()
        k = int(line.split("--")[0])
        v = int(line.split("--")[1])
        true_alignment[k] = v
        true_alignment[v] = k
    f.close()
    return true_alignment


def alignment_file_to_list(result_alignment_file):
    f = open(result_alignment_file, 'r')
    result_alignment = []
    for line in f.readlines():
        line = line.strip()
        k = int(line.split('--')[0])
        v = int(line.split('--')[1])
        result_alignment.append((k, v))
    f.close()
    return result_alignment


#根据noise随机shuffle边
def generate_noise(edge_list, symmetric_nodes=None,noise=0,  seed=7849):
    if symmetric_nodes is None:
        symmetric_nodes = set()
    edge_set=set(edge_list)
    random.seed(seed)
    num_to_shuffle=int(len(edge_set)*noise)
    node_list_a=[]
    node_list_b=[]
    while len(node_list_a)<num_to_shuffle:
        to_delete=random.choice(edge_list)
        if to_delete in edge_set and to_delete[0] not in symmetric_nodes and to_delete[1] not in symmetric_nodes:
            node_list_a.append(to_delete[0])
            node_list_b.append(to_delete[1])
            edge_set.remove(to_delete)
    random.shuffle(node_list_b)
    for i in range(len(node_list_a)):
        if node_list_a[i]>node_list_b[i]:
            t=node_list_b[i]
            node_list_b[i]=node_list_a[i]
            node_list_a[i]=t
        edge_set.add((node_list_a[i],node_list_b[i]))
    while len(edge_set)<len(edge_list):
        node_a=random.choice(edge_list)[0]
        while node_a in symmetric_nodes:
            node_a = random.choice(edge_list)[0]
        node_b = random.choice(edge_list)[1]
        while node_b in symmetric_nodes:
            node_b=random.choice(edge_list)[1]
        edge_set.add((node_a,node_b))
    return sorted(list(edge_set),key=lambda i:(i[0],i[1]))


#生成等价类长度更高的target
def generate_symmetric_nodes_plus(graph_,ratio,seed=7849):
    graph = nx.Graph()
    graph.add_nodes_from(graph_.nodes)
    graph.add_edges_from(graph_.edges)
    random.seed(seed)
    sorted_by_degree = sorted(list(graph.degree), key=lambda i: i[1])
    degree_map = {i: 0 for i in range(sorted_by_degree[-1][1] + 1)}
    for i in sorted_by_degree:
        degree_map[i[1]] += 1
    deleted_nodes = {}
    left_num_to_deleted = int(ratio * len(graph))
    graph_nodes=list(graph.nodes)
    while left_num_to_deleted > 0:
        candidate_index = int(random.random() * len(graph))
        candidate_node = graph_nodes[candidate_index]
        while candidate_node in deleted_nodes or degree_map[graph.degree[candidate_node]] == 1:
            candidate_index = int(random.random() * len(graph))
            candidate_node = graph_nodes[candidate_index]
        degree_map[graph.degree[candidate_node]] -= 1
        deleted_nodes[candidate_node] = graph.degree[candidate_node]
        left_num_to_deleted -= 1

    for i in deleted_nodes.keys():
        graph.remove_node(i)
    sorted_deleted_nodes = []
    for k, v in deleted_nodes.items():
        sorted_deleted_nodes.append((k, v))
    sorted_deleted_nodes = sorted(sorted_deleted_nodes, key=lambda i: i[1])
    template_node = None
    symmetric_nodes=set(i[0] for i in sorted_deleted_nodes)
    for i in sorted_deleted_nodes:
        if template_node is None or graph.degree[template_node]!=i[1]:
            node_list_by_degree = [j[0] for j in list(graph.degree) if j[1] == i[1]]
            template_index = int(random.random() * len(node_list_by_degree))
            template_node = node_list_by_degree[template_index]
            symmetric_nodes.add(template_node)
        graph.add_node(i[0])
        graph.add_edges_from([[i[0], j] for j in graph[template_node]])
    node_degree_zero = [i[0] for i in list(graph.degree) if i[1] == 0]
    for i in node_degree_zero:
        graph.add_edge(i, i)
    return graph,symmetric_nodes


def generate_symmetric_nodes_x(graph_,ratio):
    graph=nx.Graph()
    graph.add_nodes_from(graph_.nodes)
    graph.add_edges_from(graph_.edges)
    random.seed(7849)
    deleted_nodes = {}
    left_num_to_deleted = int(ratio * len(graph))
    while left_num_to_deleted > 0:
        candidate_index = int(random.random() * len(graph))
        candidate_node = (list(graph.nodes))[candidate_index]
        while candidate_node in deleted_nodes :
            candidate_index = int(random.random() * len(graph))
            candidate_node = (list(graph.nodes))[candidate_index]
        deleted_nodes[candidate_node] = graph.degree[candidate_node]
        left_num_to_deleted -= 1
    for i in deleted_nodes.keys():
        graph.remove_node(i)
    template_node=0
    for i in deleted_nodes.keys():
        graph.add_node(i)
        graph.add_edges_from([[i, j] for j in graph[template_node]])
    symmetric_nodes=set([i for i in deleted_nodes.keys()])
    symmetric_nodes.add(template_node)
    return graph,symmetric_nodes


def generate_symmetric_graphs(n,m,ratio):
    symme_node_num=int(n*ratio)
    symme_edge_num=int(m*ratio)
    symme_graph=nx.gnm_random_graph(symme_node_num/2,symme_edge_num/2,seed=9)
    single_nodes=[i[0] for i in list(symme_graph.degree) if i[1]==0]
    for i in single_nodes:
        symme_graph.add_edge(i,i)
    main_graph=nx.gnm_random_graph(n-symme_node_num, m-symme_edge_num, seed=9)
    single_nodes=[i[0] for i in list(main_graph.degree) if i[1]==0]
    for i in single_nodes:
        main_graph.add_edge(i,i)
    edgelist_symme_graph=list(symme_graph.edges)
    len_nonsymme_graph=len(main_graph)
    len_symme_graph=len(symme_graph)
    for i in edgelist_symme_graph:
        main_graph.add_edge(i[0]+len_nonsymme_graph,i[1]+len_nonsymme_graph)
        main_graph.add_edge(i[0]+len_nonsymme_graph+len_symme_graph,i[1]+len_nonsymme_graph+len_symme_graph)
    return main_graph


def generate_symmetric_graphs_z(n,p,ratio):
    random.seed(1)
    symmetric_num=int(ratio*n)
    if symmetric_num%2==1:
        symmetric_num+=1
    main_graph_nodes_num=n-symmetric_num/2
    main_graph=nx.gnp_random_graph(main_graph_nodes_num, p, seed=1)
    candidate_nodes=random.sample(list(main_graph.nodes),symmetric_num/2)
    candidate_nodes_duplicate={}
    for i in candidate_nodes:
        candidate_nodes_duplicate[i]=main_graph_nodes_num
        main_graph_nodes_num+=1
    candidate_neighbors=[]
    #nodes=sorted(list(main_graph.nodes),reverse=True)

    for i in candidate_nodes:
        neighbor_list=list(main_graph[i])
        candidate_neighbors.append([(i,j) for j in neighbor_list])
    for i in candidate_neighbors:
        for j in i:
            main_graph.add_edge(candidate_nodes_duplicate[j[0]],j[1])
            if j[1] in candidate_nodes:
                main_graph.add_edge(candidate_nodes_duplicate[j[0]], candidate_nodes_duplicate[j[1]])
    nodes=[i for i in list(main_graph.nodes) if len(main_graph[i])==0]
    for i in nodes:
        main_graph.add_edge(i,i)
    if len(main_graph)!=n:
        node_list=list(main_graph.nodes)
        for i in range(n):
            if i not in node_list:
                main_graph.add_edge(i,i)
    return main_graph


def generate_symmetric_nodes(graph_,ratio):
    graph=nx.Graph()
    graph.add_nodes_from(graph_.nodes)
    graph.add_edges_from(graph_.edges)
    random.seed(784)
    sorted_by_degree = sorted(list(graph.degree), key=lambda i: i[1])
    degree_map = {i: 0 for i in range(sorted_by_degree[-1][1]+1)}
    for i in sorted_by_degree:
        degree_map[i[1]] += 1
    deleted_nodes = {}
    left_num_to_deleted = int(ratio * len(graph))
    while left_num_to_deleted > 0:
        candidate_index = int(random.random() * len(graph))
        candidate_node = (list(graph.nodes))[candidate_index]
        while candidate_node in deleted_nodes or degree_map[graph.degree[candidate_node]] == 1:
            candidate_index = int(random.random() * len(graph))
            candidate_node = (list(graph.nodes))[candidate_index]
        degree_map[graph.degree[candidate_node]] -= 1
        deleted_nodes[candidate_node] = graph.degree[candidate_node]
        left_num_to_deleted -= 1

    for i in deleted_nodes.keys():
        graph.remove_node(i)
    sorted_deleted_nodes=[]
    for k,v in deleted_nodes.items():
        sorted_deleted_nodes.append((k,v))
    sorted_deleted_nodes=sorted(sorted_deleted_nodes,key=lambda i:i[1])
    for i in sorted_deleted_nodes:
        node_list_by_degree = [j[0] for j in list(graph.degree) if j[1] == i[1]]
        template_index=int(random.random() * len(node_list_by_degree))
        template_node = node_list_by_degree[template_index]
        graph.add_node(i[0])
        graph.add_edges_from([[i[0], j] for j in graph[template_node]])
    node_degree_zero = [i[0] for i in list(graph.degree) if i[1]==0]
    for i in node_degree_zero:
        graph.add_edge(i,i)
    return graph


def generate_shuffled_target(folder):
    nodes=set()
    target=open(folder+"target.txt","r")
    for line in target:
        arr = line.strip().split()
        nodes.add(arr[0])
        nodes.add(arr[1])
    target.close()
    nodes=sorted(nodes,key=lambda i:int(i))
    random.seed(1)
    shuffled_nodes=[i for i in nodes]
    random.shuffle(shuffled_nodes)
    shuffle_mapping={}
    f=open(folder+"shuffle_map.txt","w")
    for i in range(len(nodes)):
        shuffle_mapping[nodes[i]]=shuffled_nodes[i]
        f.write(nodes[i]+" "+shuffled_nodes[i]+"\n")
    f.close()
    f=open(folder+"target.txt","r")
    shuff_arr=[]
    for line in f:
        arr=line.strip().split()
        shuff_arr.append(str(shuffle_mapping[arr[0]])+" "+str(shuffle_mapping[arr[1]])+"\n")
    shuffled_target = open(folder + "shuffled_target.txt", "w")
    shuffled_target.writelines(shuff_arr)
    f.close()
    shuffled_target.close()


def to_pure_mapping(list_mapping):
    mapping = {}
    visited=set()
    for i in list_mapping:
        for k, v in i.items():
            if k in visited:
                print(":")
            mapping[k] = v
            visited.add(k)
    return mapping


def merge_mapping(automorphic_mapping_, isomorphic_mapping_):
    result_mapping = {}
    isomorphic_mapping = to_pure_mapping(isomorphic_mapping_)
    automorphic_mapping = to_pure_mapping(automorphic_mapping_)
    for key, value in automorphic_mapping.items():
        if not result_mapping.has_key(key):
            sub_set = set()
            for j in value:
                sub_set |= isomorphic_mapping[j]
            for j in sub_set:
                result_mapping[j] = sub_set
    return result_mapping

if __name__ == "__main__":
    generate_shuffled_target("Datasets/Experiment2/Node8000_M25000_Seed9/SymmeRatio000/")
    generate_shuffled_target("Datasets/Experiment2/Node8000_M25000_Seed9/SymmeRatio005/")
    generate_shuffled_target("Datasets/Experiment2/Node8000_M25000_Seed9/SymmeRatio010/")
    generate_shuffled_target("Datasets/Experiment2/Node8000_M25000_Seed9/SymmeRatio015/")
