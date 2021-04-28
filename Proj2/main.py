1`from networkx.algorithms.centrality import betweenness
from networkx.algorithms.shortest_paths.generic import shortest_path
import networkx as nx
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path
import pytest
from operator import itemgetter


g0 = nx.Graph()
g0.add_edge(1, 2)
g0.add_edge(2, 3)
g0.add_edge(2, 4)
g0.add_edge(3, 5)
g0.add_edge(4, 5)
g0.add_edge(5, 6)

graph_0 = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)]


graph_1 = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
g1 = nx.Graph()
for edge in graph_1:
    g1.add_edge(edge[0], edge[1])

graph_2 = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 1)]
g2 = nx.Graph()
for edge in graph_2:
    g2.add_edge(edge[0], edge[1])

graph_3 = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 6),
           (4, 5), (5, 1), (6, 7), (6, 8), (7, 8), (8, 9)]
g3 = nx.Graph()

for edge in graph_3:
    g3.add_edge(edge[0], edge[1])


g4 = nx.path_graph(50)
graph_4 = []
for line in nx.generate_edgelist(g4, data=False):
    vi = int(line.split()[0])
    vj = int(line.split()[1])
    pair = (vi, vj)
    graph_4.append(pair)

g5 = nx.complete_graph(7)
graph_5 = []
for line in nx.generate_edgelist(g5, data=False):
    vi = int(line.split()[0])
    vj = int(line.split()[1])
    pair = (vi, vj)
    graph_5.append(pair)

g6 = nx.cycle_graph(8)
graph_6 = []
for line in nx.generate_edgelist(g6, data=False):
    vi = int(line.split()[0])
    vj = int(line.split()[1])
    pair = (vi, vj)
    graph_6.append(pair)

n = 200  # number of nodes for Barabasi-Albert graph
q = 4  # number of edges each new node has in the generation process of BA graph

g7 = nx.barabasi_albert_graph(n, q, seed=34)
graph_7 = []
for line in nx.generate_edgelist(g7, data=False):
    vi = int(line.split()[0])
    vj = int(line.split()[1])
    pair = (vi, vj)
    graph_7.append(pair)


g8 = nx.erdos_renyi_graph(100, 0.13, seed=15)
graph_8 = []
for edge in g8.edges():
    graph_8.append(edge)


def test_numOfVertices():
    assert numOfVertices(graph_0) == 6
    assert numOfVertices(graph_1) == 5
    assert numOfVertices(graph_2) == 5
    assert numOfVertices(graph_3) == 9
    assert numOfVertices(graph_4) == 50
    assert numOfVertices(graph_5) == 7
    assert numOfVertices(graph_6) == 8
    assert numOfVertices(graph_7) == 200
    assert numOfVertices(graph_8) == 100


def test_degOfVertex():
    assert degOfVertex(graph_1, 2) == 2
    assert degOfVertex(graph_2, 2) == 3
    assert degOfVertex(graph_3, 2) == 3
    assert degOfVertex(graph_3, 3) == 4
    assert degOfVertex(graph_3, 6) == 3
    assert degOfVertex(graph_4, 25) == 2
    assert degOfVertex(graph_5, 4) == 6
    assert degOfVertex(graph_6, 3) == 2
    assert degOfVertex(graph_7, 27) == 10


def test_clusteringCoef():
    assert clusteringCoef(graph_1, 2) == 0.0
    assert clusteringCoef(graph_2, 4) == 0.3333333333333333
    assert clusteringCoef(graph_3, 3) == 0.3333333333333333
    assert clusteringCoef(graph_4, 5) == 0.0
    assert clusteringCoef(graph_5, 4) == 1.0
    assert clusteringCoef(graph_6, 1) == 0.0
    assert clusteringCoef(graph_7, 11) == 0.07142857142857142
    assert clusteringCoef(graph_8, 46) == 0.2857142857142857


def test_betweennessCent():
    assert betweennessCent(graph_0, 1) == 0.0
    assert betweennessCent(graph_0, 2) == 4.5
    assert betweennessCent(graph_0, 3) == 2.0
    assert betweennessCent(graph_0, 4) == 2.0
    assert betweennessCent(graph_0, 5) == 4.5
    assert betweennessCent(graph_0, 6) == 0.0

    assert betweennessCent(graph_1, 1) == 1.0
    assert betweennessCent(graph_1, 2) == 1.0
    assert betweennessCent(graph_1, 3) == 1.0
    assert betweennessCent(graph_1, 4) == 1.0
    assert betweennessCent(graph_1, 5) == 1.0

    assert betweennessCent(graph_6, 0) == 4.5
    assert betweennessCent(graph_6, 1) == 4.5
    assert betweennessCent(graph_6, 2) == 4.5
    assert betweennessCent(graph_6, 3) == 4.5
    assert betweennessCent(graph_6, 4) == 4.5
    assert betweennessCent(graph_6, 5) == 4.5
    assert betweennessCent(graph_6, 6) == 4.5
    assert betweennessCent(graph_6, 7) == 4.5

    assert betweennessCent(graph_7, 23) == 873.7604441701953

    assert betweennessCent(graph_8, 39) == 86.18112583907725


def test_avgShortestPath():
    assert avgShortestPath(graph_0) == 1.8666666666666667
    assert avgShortestPath(graph_2) == 1.3
    assert avgShortestPath(graph_3) == 2.25
    assert avgShortestPath(graph_4) == 17.0
    assert avgShortestPath(graph_5) == 1.0
    assert avgShortestPath(graph_6) == 2.2857142857142856
    assert avgShortestPath(graph_7) == 2.6090452261306534
    assert avgShortestPath(graph_8) == 2.0785858585858588


def test_adjMatrix():
    assert adjMatrix(graph_0) == [[0, 1, 0, 0, 0, 0, ],
                                  [1, 0, 1, 1, 0, 0, ],
                                  [0, 1, 0, 0, 1, 0, ],
                                  [0, 1, 0, 0, 1, 0, ],
                                  [0, 0, 1, 1, 0, 1, ],
                                  [0, 0, 0, 0, 1, 0, ]]

    assert adjMatrix(graph_1) == [[0, 1, 0, 0, 1, ],
                                  [1, 0, 1, 0, 0, ],
                                  [0, 1, 0, 1, 0, ],
                                  [0, 0, 1, 0, 1, ],
                                  [1, 0, 0, 1, 0, ]]

    assert adjMatrix(graph_2) == [[0, 1, 1, 0, 1, ],
                                  [1, 0, 1, 1, 0, ],
                                  [1, 1, 0, 1, 0, ],
                                  [0, 1, 1, 0, 1, ],
                                  [1, 0, 0, 1, 0, ]]

    assert adjMatrix(graph_5) == [[0, 1, 1, 1, 1, 1, 1, ],
                                  [1, 0, 1, 1, 1, 1, 1, ],
                                  [1, 1, 0, 1, 1, 1, 1, ],
                                  [1, 1, 1, 0, 1, 1, 1, ],
                                  [1, 1, 1, 1, 0, 1, 1, ],
                                  [1, 1, 1, 1, 1, 0, 1, ],
                                  [1, 1, 1, 1, 1, 1, 0, ]]

    assert adjMatrix(graph_6) == [[0, 1, 0, 0, 0, 0, 0, 1, ],
                                  [1, 0, 1, 0, 0, 0, 0, 0, ],
                                  [0, 1, 0, 1, 0, 0, 0, 0, ],
                                  [0, 0, 1, 0, 1, 0, 0, 0, ],
                                  [0, 0, 0, 1, 0, 1, 0, 0, ],
                                  [0, 0, 0, 0, 1, 0, 1, 0, ],
                                  [0, 0, 0, 0, 0, 1, 0, 1, ],
                                  [1, 0, 0, 0, 0, 0, 1, 0, ]]


def numOfVertices(list):
    # Using the built in operators we can use the itemgetter that sorts the lists of
    # tuples, so we are getting the returned tuple are largest size(x, y)
    # and then getting the largest element fom that tuple which is the
    # number of nodes in the graph
    if(min(min(list)) == 0):
        return (max(max(list, key=itemgetter(1))) + 1)
    else:
        return max(max(list, key=itemgetter(1)))


def degOfVertex(list, vertex):
    deg = 0
    for v in list:
        if vertex in v:
            deg += 1
    return deg


def clusteringCoef(list, vertex):
    graph = nx.Graph()
    links = 0
    index = 0
    deg = degOfVertex(list, vertex)
    for edge in list:
        graph.add_edge(edge[0], edge[1])

    neighbors = []
    for i in graph.edges():
        if vertex in i:
            neighbors += i
            neighbors.remove(vertex)

    if len(neighbors) <= 2:
        return 0.0

    for x in neighbors:
        index += 1
        for y in neighbors[index:]:
            if (x, y) in graph.edges():
                links += 1

    return (2.0 * links)/(deg * (deg - 1))

# Iterate over every edge


def betweennessCent(list, vertex):
    graph = nx.Graph()
    for edge in list:
        graph.add_edge(edge[0], edge[1])

    if degOfVertex(list, vertex) <= 1:
        return 0.0
    sigmaV = []
    sigma = []

    for node in graph.nodes():
        paths = all_pairs_shortest_path(graph, node)
        for v in paths:
            if len(v[1].values()) > 2 and v[0] != vertex:
                print(v[1].values())
            # betweenness = 0
            # betweennessDict = {}
            # pathDict = nx.all_simple_edge_paths(graph)

            # for s in graph.nodes:
            #     for i in graph.nodes:
            #         sigmaV = 0
            #         if path in pathDict[i]:
            #             if s in path and s not in i:
            #                 sigmaV += 1

            # print(sigmaV)

            # betweenness += len(sigma)/sigmaV

    return betweenness


# %%


def adjMatrix(list):
    graph = nx.Graph()

    for edge in list:
        graph.add_edge(edge[0], edge[1])

    nodes = len(graph.nodes())
    adjMat = [[0 for _ in range(nodes)] for _ in range(nodes)]
    allEdges = []

    for list in graph.edges():
        for node in graph.edges(list):
            if node not in allEdges:
                allEdges.append(node)
            if tuple(reversed(node)) not in allEdges:
                x = tuple(reversed(node))
                allEdges.append(x)

    xEdge = []
    yEdge = []

    for i in range(len(allEdges)):
        xEdge.append(allEdges[i][0])
        yEdge.append(allEdges[i][1])

    for i in range(len(allEdges)):
        x = xEdge[i] - 1
        y = yEdge[i] - 1
        adjMat[x][y] = 1

    return adjMat


def avgShortestPath(list):
    graph = nx.Graph()

    for edge in list:
        graph.add_edge(edge[0], edge[1])
    nodes = len(graph.nodes())
    lenOF = 0
    for node in graph.nodes():
        paths = shortest_path(graph, node)
        for v in paths.values():
            if len(v) > 1:
                lenOF += len(v) - 1
    return ((1 / (nodes * (nodes - 1))) * lenOF)


test_numOfVertices()
test_degOfVertex()
test_clusteringCoef()
test_avgShortestPath()
test_adjMatrix()
test_betweennessCent()



