import json
import networkx as nx
from networkx.readwrite import json_graph

with open('../dataset/graph_v1.json', 'r') as rf:
    data = json.loads(rf.readline())
    G = json_graph.node_link_graph(data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            broken_count += 1
            G.remove_node(node)
    print(broken_count)
