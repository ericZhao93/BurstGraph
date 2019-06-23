from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
print(version_info)
major = version_info[0]
minor = version_info[1]
#assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=4
N_WALKS=10

def load_data(prefix, normalize=True, load_walks=False, split_class=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(list(G.nodes())[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    
    broken_count = 0
    pred_nodes = list(G.nodes())
    print(len(pred_nodes))
    for node in pred_nodes:
        if ('test' not in G.node[node]) and (not 'user' in G.node[node]):
            G.remove_node(node)
            broken_count += 1
        #break
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    
    class_map = {}
    missed_class_set = set([])
    for node in G.nodes():
        if not G.node[node]["user"]:
            class_map[node] = len(class_map)
    print("{:d} categories in the network".format(len(class_map)))
    print(class_map)
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    train_class_map = json.load(open(prefix + "-train.json"))
    all_class = {}
    if not split_class:
        train_degrees = 0.0
        for node, category in train_class_map.items():
            vec = [0.0 for i in range(len(class_map))]
            for cat in category["new_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec[class_map[cat]] = 1.0

            for cat in category["old_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec[class_map[cat]] = 1.0
            all_class[node] = vec[:]
            train_degrees += sum(vec)
        print("average categorys: {:.3f} in train dataset".format(train_degrees / len(train_class_map.items())))
    else:
        train_old_degrees , train_new_degrees = 0.0, 0.0
        for node, category in train_class_map.items():
            vec = [0.0 for i in range(len(class_map))]
            for cat in category["old_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec[class_map[cat]] = 1.0
            vec1 = [0.0 for i in range(len(class_map))]
            for cat in category["new_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec1[class_map[cat]] = 1.0
            all_class[node] = {'old': vec[:], 'new': vec1[:]}
            train_old_degrees += sum(vec)
            train_new_degrees += sum(vec1)
        print("average old categorys: {:.3f} , new categorys: {:.3f} in train dataset".format(train_old_degrees / len(train_class_map), train_new_degrees / len(train_class_map)))


    test_class_map = json.load(open(prefix + "-test.json"))
    test_class = {}
    if not split_class:
        test_degrees = 0.0
        for node, category in test_class_map.items():
            vec = category['old_cate'][:]
            vec.extend(category['new_cate'][:])
            test_class[node] = { 'pos': vec[:], 'neg': category['neg_cate'][:]}
            vec = [0.0 for i in range(len(class_map))]
            for cat in category["new_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec[class_map[cat]] = 1.0
            for cat in category["old_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec[class_map[cat]] = 1.0
            all_class[node] = vec[:]
            test_degrees += sum(vec)
        print("average categorys: {:.3f} in test dataset".format(test_degrees / len(test_class_map)))
    else:
        test_old_degrees , test_new_degrees = 0.0, 0.0
        for node, category in test_class_map.items():
            test_class[node] = {'old': category["old_cate"][:], 'new': category["new_cate"][:], 'neg': category["neg_cate"][:]}
            vec = [0.0 for i in range(len(class_map))]
            for cat in category["old_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec[class_map[cat]] = 1.0
            vec1 = [0.0 for i in range(len(class_map))]
            for cat in category["new_cate"]:
                if cat not in class_map:
                    missed_class_set.add(cat)
                else:
                    vec1[class_map[cat]] = 1.0
            all_class[node] = {'old': vec[:], 'new': vec1[:]}
            test_old_degrees += sum(vec)
            test_new_degrees += sum(vec1)
        print("average old categorys: {:.3f}, new categorys: {:.3f} in test dataset".format(test_old_degrees / len(test_class_map), test_new_degrees / len(test_class_map)))
    print("missed class set length: {:d}".format(len(missed_class_set)))
    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if  ((not G.node[edge[0]]['user'] or G.node[edge[0]]['test']) and (not G.node[edge[1]]['user'] or G.node[edge[1]]['test'])):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['test'] and G.node[n]['user']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, all_class, class_map, test_class


def load_seq_data(prefix, num_steps=5, normalize=True, load_walks=False, split_class=False):
    Gs = []
    all_results, test_results = [], []
    id_map = json.load(open(prefix+'alibaba_id_map.json'))
    id_map = {k: int(v) for k,v in id_map.items()}
    class_map = {}
    for step in range(num_steps):
        G_data = json.load(open(prefix + "graph/alibaba_gul_graph_{:d}.json".format(13+step)))
        G = json_graph.node_link_graph(G_data)
        num = 0
        for node in G.nodes():
            if not G.node[node]["user"] and node not in class_map:
                class_map[node] = len(class_map)
            if not G.node[node]["user"]:
                num += 1
        print("{:d} categories in the graph {:d}".format(num, step))
        print("class map length: {:d}".format(len(class_map)))
    print(class_map)
    for step in range(num_steps):
        G_data = json.load(open(prefix + "graph/alibaba_gul_graph_{:d}.json".format(13+step)))
        G = json_graph.node_link_graph(G_data)
        if isinstance(list(G.nodes())[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n

        if os.path.exists(prefix + "features.npy"):
            feats = np.load(prefix + "features.npy")
        else:
            print("No features present.. Only identity features will be used.")
            feats = None
    
        broken_count = 0
        pred_nodes = list(G.nodes())
        print("total node numbers in graph {:d} is {:d}".format(step, len(pred_nodes)))
        for node in pred_nodes:
            if ('test' not in G.node[node]) and (not 'user' in G.node[node]):
                G.remove_node(node)
                broken_count += 1
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    
        missed_class_set = set([])
        walks = []
        train_class_map = json.load(open(prefix + "label/alibaba_gul_graph_train_label_{:d}.json".format(step+14)))
        all_class = {}
        if not split_class:
            train_degrees = 0.0
            for node, category in train_class_map.items():
                vec = [0.0 for i in range(len(class_map))]
                for cat in category["new_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec[class_map[cat]] = 1.0

                for cat in category["old_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec[class_map[cat]] = 1.0
                all_class[node] = vec[:]
                train_degrees += sum(vec)
            print("average categorys: {:.3f} in train dataset".format(train_degrees / len(train_class_map.items())))
        else:
            train_old_degrees , train_new_degrees = 0.0, 0.0
            for node, category in train_class_map.items():
                vec = [0.0 for i in range(len(class_map))]
                for cat in category["old_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec[class_map[cat]] = 1.0
                vec1 = [0.0 for i in range(len(class_map))]
                for cat in category["new_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec1[class_map[cat]] = 1.0
                all_class[node] = {'old': vec[:], 'new': vec1[:]}
                train_old_degrees += sum(vec)
                train_new_degrees += sum(vec1)
            print("average old categorys: {:.3f} , new categorys: {:.3f} in train dataset".format(train_old_degrees / len(train_class_map), train_new_degrees / len(train_class_map)))


        test_class_map = json.load(open(prefix + "label/alibaba_gul_graph_test_label_{:d}.json".format(step+14)))
        test_class = {}
        if not split_class:
            test_degrees = 0.0
            for node, category in test_class_map.items():
                vec = category['old_cate'][:]
                vec.extend(category['new_cate'][:])
                test_class[node] = { 'pos': vec[:], 'neg': category['neg_cate'][:]}
                vec = [0.0 for i in range(len(class_map))]
                for cat in category["new_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec[class_map[cat]] = 1.0
                for cat in category["old_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec[class_map[cat]] = 1.0
                all_class[node] = vec[:]
                test_degrees += sum(vec)
            print("average categorys: {:.3f} in test dataset".format(test_degrees / len(test_class_map)))
        else:
            test_old_degrees , test_new_degrees = 0.0, 0.0
            for node, category in test_class_map.items():
                test_class[node] = {'old': category["old_cate"][:], 'new': category["new_cate"][:], 'neg': category["neg_cate"][:]}
                vec = [0.0 for i in range(len(class_map))]
                for cat in category["old_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec[class_map[cat]] = 1.0
                vec1 = [0.0 for i in range(len(class_map))]
                for cat in category["new_cate"]:
                    if cat not in class_map:
                        missed_class_set.add(cat)
                    else:
                        vec1[class_map[cat]] = 1.0
                all_class[node] = {'old': vec[:], 'new': vec1[:]}
                test_old_degrees += sum(vec)
                test_new_degrees += sum(vec1)
            print("average old categorys: {:.3f}, new categorys: {:.3f} in test dataset".format(test_old_degrees / len(test_class_map), test_new_degrees / len(test_class_map)))
        print("missed class set length: {:d}".format(len(missed_class_set)))
        ## Make sure the graph has edge train_removed annotations
        ## (some datasets might already have this..)
        print("Loaded data.. now preprocessing..")
        for edge in G.edges():
            if  ((not G.node[edge[0]]['user'] or G.node[edge[0]]['test']) and (not G.node[edge[1]]['user'] or G.node[edge[1]]['test'])):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        if normalize and not feats is None:
            from sklearn.preprocessing import StandardScaler
            train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['test'] and G.node[n]['user']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)
    
        if load_walks:
            with open(prefix + "-walks.txt") as fp:
                for line in fp:
                    walks.append(map(conversion, line.split()))
        Gs.append(G)
        all_results.append(all_class)
        test_results.append(test_class)
    return Gs, feats, id_map, walks, all_results, class_map, test_results



def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                try:
                    next_node = random.choice(list(G.neighbors(curr_node)))
                    # self co-occurrences are useless
                    if curr_node != node:
                        pairs.append((node,curr_node))
                    curr_node = next_node
                except:
                    print(len(G.neighbors(curr_node)))
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    load_seq_data("../sequential/")
    
    """ Run random walks """
    '''
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    broken_count = 0
    pre_list = list(G.nodes())
    for n in pre_list:
        if not "user" in G.node[n] or not "test" in G.node[n]:
            broken_count+=1
            G.remove_node(n)
    item_nodes = []
    for n in G.nodes():
        if "user" in G.node[n] and not G.node[n]["user"]:
            item_nodes.append(n)
    print(len(set(item_nodes)))
    remove_nodes = []
    for n in G.nodes():
        if "test" not in G.node[n]:
            #print(n) 
            remove_nodes.append(n)
    print(len(set(remove_nodes)))
    print(list(G.nodes())[:10])
    
    print(list(G.neighbors(1)))
    train_users_and_items = [n for n in G.nodes() if (G.node[n]["user"] and not G.node[n]["test"]) or (not G.node[n]["user"])]      
    nodes = [n for n in G.nodes() if G.node[n]["user"] and not G.node[n]["test"]]
    G = G.subgraph(train_users_and_items)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
    '''
