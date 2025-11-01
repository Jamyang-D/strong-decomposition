# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:34:06 2024

@author: dyx123
"""

import os
import copy
import random
import time
from itertools import chain, combinations
import webbrowser
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pgmpy
import torch
import pprint
from IPython.display import Image
from pgmpy.utils import get_example_model
import numpy as np

# Randomly generate a connected CG
def generate_random_lwf(n, edge_density):
    """
    Input:
        n: int - Number of nodes in the CG.
        edge_density: float - Probability of edge creation between any two nodes.

    Output:
        graph: DiGraph - A directed acyclic graph (CG) generated randomly.
    """

    graph = nx.DiGraph()
    nodes = ['v' + str(i) for i in range(1, n + 1)]
    for node in nodes:
        graph.add_node(node)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_density:
                graph.add_edge(nodes[i], nodes[j])
    return graph


# Check connected components in a CG
def connect_components(graph):
    """
    Input:
        graph: DiGraph - A directed acyclic graph (CG).

    Output:
        None (modifies the input graph by connecting components if needed).
    """

    components = list(nx.weakly_connected_components(graph))
    num_components = len(components)
    if num_components > 1:
        for i in range(num_components - 1):
            root1 = next(iter(components[i]))
            root2 = next(iter(components[i + 1]))
            graph.add_edge(root1, root2)


## Randomly generate a connected CG
def Generate_lwf(n, edge_density, p_undirected):
    """
    Input:
        n: int - Number of nodes in the CG.
        edge_density: float - Probability of edge creation between any two nodes.

    Output:
        graph: dict - A connected CG represented as a dictionary.
    """

    random_cg = generate_random_lwf(n, edge_density)
    connect_components(random_cg)
    graph = {node: {} for node in random_cg.nodes}
    for node in random_cg.nodes:
        neighbors = list(random_cg.neighbors(node))
        neighbors = [neighbor for neighbor in neighbors if neighbor != node]
        graph[node] = {neighbor: 'b' for neighbor in neighbors}
    # 单向边变双向边
    for u in graph.keys():
        for v in graph[u].keys():
            if random.random() < p_undirected:
                graph[v][u] = graph[u][v]
    return graph


#########################################
# Convert CG format
def convert_cg(cg_graph):  # dictionary to list
    """
    Input:
        cg_graph: dict - A CG represented as a dictionary.  CG={1:{2:'b',3:'b'},2:{3:'b'}}

    Output:
        new_cg_graph: dict - A CG represented as a  list. CG={1:[2,3],2:[3]}
    """
    new_cg_graph = {node: {} for node in cg_graph.keys()}
    for node, neighbors in cg_graph.items():
        new_cg_graph[node] = list(neighbors.keys())
    return new_cg_graph


################
# Output the reversed graoh
def reverse_graph(graph):
    """
    Input:
        graph: dict - A CG represented as a dictionary.

    Output:
        reversed_graph: dict - The reversed CG, where all edges are reversed.
    """

    reversed_graph = {node: {} for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in reversed_graph:
                reversed_graph[neighbor] = {}
            reversed_graph[neighbor][node] = graph[node][neighbor]
    return reversed_graph


# Plot
def plot_result(result, file_path, title=None):
    """
    Input:
        result: dict - A CG represented as a dictionary.
        file_path: str - The path where the plot will be saved.
        title: str or None - The title of the plot.

    Output:
        None (saves a plot of the CG to the specified file path).
    """
    G = nx.DiGraph()
    # Add nodes
    for node in result:
        G.add_node(node)
        # Add edges and annotation information

    edge_colors = []
    for node, neighbors in result.items():
        for neighbor, info in neighbors.items():
            G.add_edge(node, neighbor)
            edge_colors.append(info)
            # plot
    pos = nx.circular_layout(G)  # Use the circular_layout layout method
    # Create an image of size 8x6 inches with dpi set to 150
    plt.figure(figsize=(16, 12), dpi=150)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue',
                     node_size=2000, font_size=28, edge_color=edge_colors)

    if title:
        plt.title(title)
    else:
        plt.title("initial")

    plt.savefig(file_path)
    plt.show()
    # plt.close()


#########################
# Children set of a vertex
def ch(CG, vertices, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        vertices: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of children of the given vertex.
    """
    CH = set()
    for v in CG[vertices].keys():
        if vertices not in CG[v].keys():
            CH.add(v)
    return CH


# Parent set of  a vertex
def pa(CG, vertices, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        vertices: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of parents of the given vertex.
    """
    PA = set()
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    for v in reverse_CG[vertices].keys():
        if vertices not in reverse_CG[v].keys():
            PA.add(v)
    return PA


def ne(CG, vertices, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        node: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of neighbors of the given vertex.
    """
    NE = set()
    for v in CG[vertices].keys():
        if vertices in CG[v].keys():
            NE.add(v)
    return NE


def tch(CG, vertices, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        vertices: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of spouses of the given vertex.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    ch1 = set()
    ch_add = ch(CG, vertices, reverse_CG)
    ch1 = ch1 | ch_add
    while ch_add:
        ch_add1 = set()
        for v in ch_add:
            ch_add1 = ch_add1 | ne(CG, v, reverse_CG)
        ch_add = ch_add1 - ch1
        ch1 = ch1 | ch_add
    tch_value = set()
    for i in ch1:
        tch_value = tch_value.union(pa(CG, i, reverse_CG))
    tch_value.discard(vertices)
    # tch_value = tch_value - ch1
    return tch_value


# Spouse set of vertices
def set_tch(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of spouses of the given vertices.
    """
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    tch_set = set()
    for i in nodes:
        tch_set = tch_set | tch(CG, i, reverse_CG)
    tch_set = tch_set - set(nodes)
    return tch_set


# Children set of vertices
def set_ch(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.

    Output:
        set - The set of children of the given vertices.
    """
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    CHS = set()
    for i in nodes:
        CHS = CHS | ch(CG, i, reverse_CG)
    CHS = CHS - set(nodes)
    return CHS


# Parent set of vertices
def set_pa(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of parents of the given vertices.
    """
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    PAS = set()
    for i in nodes:
        PAS = PAS | pa(CG, i, reverse_CG)
    PAS = PAS - set(nodes)
    return PAS


# Neighbor set of vertices
def set_ne(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of neighbors of the given vertices.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    NES = set()
    for i in nodes:
        NES = NES | ne(CG, i, reverse_CG)
    NES = NES - set(nodes)
    return NES


# Ancestor set of vertices
def ant(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of ancestors of the given vertices.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    visited = set()
    stack = list(nodes)
    ancestors = set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            ancestors.add(node)
            stack.extend(reverse_CG.get(node, {}).keys())
    return ancestors


# Markov blanket of  a vertex
def markov_blanket(CG, vertices, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.

    Output:
        set - The set of markov blanket of a given vertex.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    pa_ = pa(CG, vertices, reverse_CG)
    ch_ = ch(CG, vertices, reverse_CG)
    ne_ = ne(CG, vertices, reverse_CG)
    tch_ = tch(CG, vertices, reverse_CG)
    markov = ch_ | pa_ | tch_ | ne_
    return markov


# Markov blanket of  vertices
def set_markov_blanket(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.

    Output:
        set - The set of markov blanket of  given vertices.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    set_pa_ = set_pa(CG, nodes, reverse_CG)
    set_ch_ = set_ch(CG, nodes, reverse_CG)
    set_ne_ = set_ne(CG, nodes, reverse_CG)
    set_tch_ = set_tch(CG, nodes, reverse_CG)
    set_markov = set_ch_ | set_pa_ | set_ne_ | set_tch_
    return set_markov


# G1-G2
def dict_difference(dict1, dict2):
    """
    Calculate the difference between two dictionaries.

    Input:
    - dict1: The first dictionary.
    - dict2: The second dictionary to compare with the first.

    Output:
    - difference: A dictionary containing the elements in dict1 that are not in dict2 or that differ between dict1 and dict2.
    """

    difference = {node: {} for node in dict1.keys()}
    for key in dict1:
        if key not in dict2:
            difference[key] = dict1[key]
        else:
            sub_dict = {}
            for sub_key in dict1[key]:
                if sub_key not in dict2[key] or dict1[key][sub_key] != dict2[key][sub_key]:
                    sub_dict[sub_key] = dict1[key][sub_key]
            if sub_dict:
                difference[key] = sub_dict
    return difference


###################

# Moralization on a CG
def moralize_cg(CG, reverse_CG=None):
    """
    Moralize a CG, connecting all parents of each node.

    Input:
    - cg: The original CG.
    - cg_reverse: (Optional) The reversed CG. If not provided, it will be computed.

    Output:
    - moralized_cg: The moralized CG.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    CG_add = dict()
    nodes = list(CG.keys())
    for parents_i in nodes:
        tch_v = tch(CG, parents_i, reverse_CG)
        for parents_j in tch_v:
            if parents_i not in CG_add:
                CG_add[parents_i] = {}
            if parents_j not in CG_add:
                CG_add[parents_j] = {}
            CG_add[parents_i][parents_j] = 'b'
            CG_add[parents_j][parents_i] = 'b'
    moralized_cg = dict()
    graphs_to_merge = [CG, CG_add, reverse_CG]
    for graph in graphs_to_merge:
        for node, neighbors in graph.items():
            moralized_cg.setdefault(node, {}).update(neighbors)
    return moralized_cg


# Obtain a induced graph from a CG
def sub_cg(CG, H):
    """
    Obtain a subgraph induced by a set of vertices from the CG.

    Input:
    - CG: The original CG.
    - H: A set of vertices to induce the subgraph.

    Output:
    - sub_CG: The induced subgraph.
    """

    sub_CG = dict()
    for i, j_dict in CG.items():
        if i in H:
            sub_CG[i] = dict()
            for j, value in j_dict.items():
                if j in H:
                    sub_CG[i][j] = value
    return sub_CG


# Determine if two vertices are connected in a UG
def is_connected(graph, start, target):
    """
    Determine if two vertices are connected in an undirected graph.

    Input:
    - graph: The undirected graph.
    - start: The starting vertex.
    - target: The target vertex.

    Output:
    - Boolean value indicating whether there is a path connecting the start and target vertices.
    """

    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        visited.add(node)
        if node == target:
            return True
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                stack.append(neighbor)
    return False


# Check if there exists an inducing path between two vertices in a CG
def is_inducing_path(CG, u, v, M, reverse_CG=None, re_MF=False):
    """
    Check if there exists an inducing path between two vertices in a CG.

    Input:
    - CG: The original CG.
    - u: The first vertex.
    - v: The second vertex.
    - M: A set of vertices to consider for the inducing path.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.
    - re_MF: (Optional) A boolean to determine if the function should return additional details.   
    Output:
    - Boolean value indicating whether there exists an inducing path between u and v.
      If re_MF is True, a tuple containing the vertices and the boolean value is returned.
      """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    nodes = [u, v]
    ANT = ant(CG, nodes, reverse_CG)
    CG_ANT = sub_cg(CG, list(ANT))  # ancestral graph
    MCG_ANT = moralize_cg(CG_ANT)  # ancestral graph after moralizing
    MCG_M_u_v = sub_cg(MCG_ANT, M + nodes)  # E
    flag = is_connected(MCG_M_u_v, u, v)  # V+E
    return (nodes, flag) if re_MF else flag


# Check t-removability
def ITRSA1_(CG, M, R1=None, reverse_CG=None):
    """
    Check t-removability of a set of vertices in a CG and return the flag and list of inducing paths.

    Input:
    - CG: The original CG.
    - M: The set of vertices to check for t-removability.
    - R1: (Optional) The set of vertices not in M. If not provided, it will be computed.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.

    Output:
    - A tuple containing a boolean indicating t-removability and a list of mf-pairs .
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    if R1 is None:
        R1 = [x for x in CG if x not in M]
    MF = list()
    f = True
    for i in range(1, len(R1)):
        for j in range(i):  #
            if R1[j] not in CG[R1[i]] and R1[i] not in CG[R1[j]]:
                nodes, flag = is_inducing_path(CG, R1[i], R1[j], M, reverse_CG, re_MF=True)
                if flag:
                    f = False
                    MF.append(nodes)
    return f, MF


# Check t-removability
def ITRSA_(CG, M, R1=None, reverse_CG=None):
    """
    Check t-removability of a set of vertices in a CG.

    Input:
    - CG: The original CG.
    - M: The set of vertices to check for t-removability.
    - R1: (Optional) The set of vertices not in M. If not provided, it will be computed.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.

    Output:
    - Boolean value indicating t-removability.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    if R1 is None:
        R1 = [x for x in CG if x not in M]
    for i in range(1, len(R1)):
        for j in range(i):  # V^2
            if R1[j] not in CG[R1[i]] and R1[i] not in CG[R1[j]]:
                if is_inducing_path(CG, R1[i], R1[j], M, reverse_CG):
                    return False
    return True


# Check t-removability
def ITRSA(CG, M, reverse_CG=None, re_MF=False):
    """
    Check the t-removability of a set of vertices.

    Input:
    - CG: The original Directed Acyclic Graph (CG).
    - M: The set of vertices to check for t-removability.
    - re_MF: (Optional) If True, returns detailed information about the mf-pairs; 
             otherwise, returns only a boolean value.

    Output:
    - If re_MF is True, returns a tuple with a boolean indicating t-removability and a list of mf-pairs;
      otherwise, returns a boolean indicating whether the set of vertices is t-removable.
    """
    R = [x for x in CG if x not in M]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    set_markov = set_markov_blanket(CG, M, reverse_CG)
    R1 = [node for node in R if node in set_markov]
    if re_MF:
        return ITRSA1_(CG, M, R1, reverse_CG)
    else:
        return ITRSA_(CG, M, R1, reverse_CG)


def c_Proximal_separator(CG, u, v, reverse_CG=None):
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    nodes = [u, v]
    ANT = ant(CG, nodes, reverse_CG)
    CG_ANT = sub_cg(CG, ANT)  # ancestral graph
    MCG_ANT = moralize_cg(CG_ANT)  # ancestral graph after moralizing
    u_ne = set(MCG_ANT[u].keys())
    c_p_s = set()
    for node in u_ne:
        sub = ANT - {u, } - u_ne | {node, }
        G_sub = sub_cg(MCG_ANT, sub)
        if is_connected(G_sub, node, v):
            c_p_s.add(node)
    return c_p_s


# Check if vertices of pa(w) are adjacent unless both of them belongs to R
def pd(G, Ma, pa1):  ##########################
    """
    Check if the parent nodes of a vertex in the graph are adjacent, unless both belong to the set R.

    Input:
    - G: The original graph.
    - Ma: The set of vertices to check.
    - pa1: The set of parent nodes of Ma.

    Output:
    - Returns True if the parent nodes are adjacent or if both belong to R; otherwise, returns False.
    """
    PA = set()
    Ma = list(Ma)
    # pa1 = list(pa1)
    for i in range(1, len(Ma)):
        for j in range(i):
            if Ma[i] not in pa1 or Ma[j] not in pa1:
                if Ma[i] not in G[Ma[j]] and Ma[j] not in G[Ma[i]]:
                    PA.add(Ma[i])
                    PA.add(Ma[j])
    PA -= pa1
    return PA


def condition(DAG, M, R, reverse_DAG=None):  #############################
    """
    Check the condition for c-removability in a Directed Acyclic Graph (DAG).

    Input:
    - DAG: The original Directed Acyclic Graph (DAG).
    - M: The set of vertices to check for c-removability.
    - R: The set of remaining vertices after excluding M.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.

    Output:
    - Returns True if the condition for c-removability is met; otherwise, returns False.
    """
    PA = set()
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    AN = ant(DAG, R, reverse_DAG)
    CH = set_ch(DAG, M, reverse_DAG) | set(M)
    w = AN & CH
    for i in w:  # V
        paa = pa(DAG, i, reverse_DAG)
        PA |= pd(DAG, paa, paa & set(R))
    return PA


# Check  c-removability
def is_cremoved(G, vertices, reverse_DAG=None):
    """
    Check if a vertex is c-removable from the DAG.
    
    Input:
    - G: The original Directed Acyclic Graph (DAG).
    - vertices: The vertex to check for c-removability.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.
    
    Output:
    - Boolean value indicating whether the vertex is c-removable.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(G)
    Ma = list(markov_blanket(G, vertices, reverse_DAG)) + [vertices]
    pa1 = list(pa(G, vertices, reverse_DAG))
    for i in range(1, len(Ma)):
        for j in range(i):
            if Ma[i] not in pa1 or Ma[j] not in pa1:
                if Ma[i] not in G[Ma[j]].keys() and Ma[j] not in G[Ma[i]].keys():
                    return False
    return True


# Check sequentially c-removability
def is_set_cremoved(G, M):
    """
    Check if a set of vertices is sequentially c-removable from the DAG.
    
    Input:
    - G: The original DAG.
    - M: A list of vertices to check for sequential c-removability.
    
    Output:
    - A tuple containing a boolean indicating if the entire set is c-removable and a list of c-removable vertices in sequence.
    """
    G1 = copy.deepcopy(G)
    G2 = reverse_graph(G1)
    M1 = copy.deepcopy(M)
    k = 1
    M_T = []
    while M1:
        del1 = {}
        for i in M1:
            if is_cremoved(G1, i, G2):
                M_T.append(i)
                M1.remove(i)
                del1[i] = G1[i]
                for j in G2[i].keys():
                    if j not in del1:
                        del1[j] = {}
                    del1[j][i] = G2[i][j]
                G1 = dict_difference(G1, del1)
                G2 = dict_difference(G2, reverse_graph(del1))
                break
        if k != len(M_T):
            break
        else:
            k = k + 1
    return len(M_T) == len(M), M_T

#Finding the c-convex hull via minimal separators
def CMCSA111(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    _, Q = ITRSA(CG, M, reverse_CG, re_MF=True)
    PA = condition(CG, M, R, reverse_CG)
    H = copy.deepcopy(R)
    while Q or PA:
        for u, v in Q:
            u_c = c_Proximal_separator(CG, u, v, reverse_CG=reverse_CG)
            v_c = c_Proximal_separator(CG, u, v, reverse_CG=reverse_CG)
            H |= u_c | v_c
        H |= PA
        M = [x for x in CG if x not in H]
        _, Q = ITRSA(CG, M, reverse_CG, re_MF=True)
        PA = condition(CG, M, H, reverse_CG)
    return H

#Find an inducing path between u and v
def uv_DCL(CG, u, v, M, reverse_CG=None):
    """
    Check if there exists an inducing path between two vertices in a CG.

    Input:
    - CG: The original CG.
    - u: The first vertex.
    - v: The second vertex.
    - M: A set of vertices to consider for the inducing path.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.
    - re_MF: (Optional) A boolean to determine if the function should return additional details.
    Output:
    - Boolean value indicating whether there exists an inducing path between u and v.
      If re_MF is True, a tuple containing the vertices and the boolean value is returned.
      """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    nodes = [u, v]
    ANT = ant(CG, nodes, reverse_CG)
    CG_ANT = sub_cg(CG, list(ANT))  # ancestral graph
    MCG_ANT = moralize_cg(CG_ANT)  # ancestral graph after moralizing
    MCG_M_u_v = sub_cg(MCG_ANT, M + nodes)  # E

    DCL = BFS_shortest_path(MCG_M_u_v, u, v)  # V+E
    return DCL

#finding all inducing paths in R
def all_DCL(CG, M, reverse_CG=None):
    """
    Check the t-removability of a set of vertices.

    Input:
    - CG: The original Directed Acyclic Graph (CG).
    - M: The set of vertices to check for t-removability.
    - re_MF: (Optional) If True, returns detailed information about the mf-pairs;
             otherwise, returns only a boolean value.

    Output:
    - If re_MF is True, returns a tuple with a boolean indicating t-removability and a list of mf-pairs;
      otherwise, returns a boolean indicating whether the set of vertices is t-removable.
    """
    R = [x for x in CG if x not in M]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    set_markov = set_markov_blanket(CG, M, reverse_CG)
    R1 = [node for node in R if node in set_markov]

    all_D = set()
    for i in range(1, len(R1)):
        for j in range(i):  #
            if R1[j] not in CG[R1[i]] and R1[i] not in CG[R1[j]]:
                DLC = uv_DCL(CG, R1[i], R1[j], M, reverse_CG)
                all_D |= set(DLC)
    return all_D

#FINDING T CONVEX HULL VIA ABSORBING INDUCING PATHS
def CMCSA_new(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    All_D = all_DCL(CG, M, reverse_CG)
    H = copy.deepcopy(R)
    while All_D:
        H |= All_D
        M = [x for x in CG if x not in H]
        All_D = all_DCL(CG, M, reverse_CG)
    return H

#FINDING C CONVEX HULL VIA ABSORBING INDUCING PATHS
def CMCSA111_new(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    All_D = all_DCL(CG, M, reverse_CG)
    PA = condition(CG, M, R, reverse_CG)
    H = copy.deepcopy(R)
    while All_D or PA:
        H |= All_D
        H |= PA
        M = [x for x in CG if x not in H]
        All_D = all_DCL(CG, M, reverse_CG)
        PA = condition(CG, M, H, reverse_CG)
    return H

def BFS_shortest_path(graph, start, goal):  # Find the shortest path using BFS
    # Initialize a queue to store the current node being explored and its path
    queue = [(start, [start])]

    # Initialize a set to keep track of visited nodes
    visited = set()

    while queue:
        # Dequeue the first element
        (vertex, path) = queue.pop(0)

        # If the current node has not been visited yet
        if vertex not in visited:
            # Mark it as visited
            visited.add(vertex)

            # Iterate over all adjacent nodes of the current node
            for next_vertex in graph[vertex]:
                # If the target node is found
                if next_vertex == goal:
                    # Return the complete path to the goal node
                    return path + [next_vertex]

                # Otherwise, enqueue the adjacent node along with its path for further exploration
                else:
                    queue.append((next_vertex, path + [next_vertex]))

    # If the queue is exhausted and the goal node is not found, return an empty list
    # indicating that no such path exists
    return []

#####################
def in_degrees(CG, reverse_CG=None):  # Indegree of a DAG
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    in_dag = dict()
    for node in reverse_CG:
        in_dag[node] = len(reverse_CG[node])
    return in_dag

def topoSort(CG, reverse_CG=None):  # Topological sort for a DAG
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)

    in_dag = in_degrees(CG, reverse_CG)

    Q = [u for u in in_dag if in_dag[u] == 0]  # Select vertices with in-degree zero
    Seq = []

    while Q:
        u = Q.pop()  # Remove the last element by default (acts as a stack)
        Seq.append(u)
        for v in CG[u]:
            in_dag[v] -= 1  # Remove the incoming edge to child node v
            if in_dag[v] == 0:
                Q.append(v)  # Add newly zero in-degree vertices for the next iteration

    if len(Seq) == len(CG):  # Check if the number of output vertices equals the total number of vertices
        return Seq
    else:
        print("G is not a directed acyclic graph.")
        return None

# C-DECOMPOSITION
def EDC(CG, reverse_CG=None):  
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    topo = topoSort(CG, reverse_CG)
    D_G = []
    for v in topo[::-1]:
        #v = topo.pop(-1)
        PA = pa(CG, v, reverse_CG)
        fa = PA | {v}
        if all(not fa.issubset(h) for h in D_G):
            H = CMCSA111_new(CG, fa, reverse_CG)
            D_G.append(H)
    return D_G




