import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
import networkx as nx


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    def _is_connected(adj_mat: np.ndarray, mst: np.ndarray):
        """
        
        Check if the MST is connected.
        
        """
        #check that all nodes in the MST are connected
        visited = [False for _ in range(len(mst))]
        visited[0] = True
        stack = [0]
        while stack:
            node = stack.pop()
            for i in range(len(mst)):
                if mst[node][i] > 0 and not visited[i]:
                    visited[i] = True
                    stack.append(i)
        return all(visited)
    assert _is_connected(adj_mat, mst), 'Proposed MST is not connected'


    # Check for the correct number of edges in the MST
    num_edges = np.count_nonzero(mst) / 2  # Divide by 2 because the matrix is symmetric
    assert num_edges == len(mst) - 1, f'Proposed MST has incorrect number of edges: {num_edges}' #comparing to number of nodes - 1



def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    allowed_error = 0.005
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()

    # Create a networkx graph for comparison
    G = nx.from_numpy_array(dist_mat)
    known_mst = nx.minimum_spanning_tree(G, weight='weight')
    known_mst_weight = known_mst.size(weight='weight')

    # Check that the sum of the edges is within the allowed error margin (0.5% of total weights from networkx MST)
    my_mst_weight = np.sum(g.mst) / 2  # Assuming that the MST is symmetric
    assert abs(my_mst_weight - known_mst_weight) <= allowed_error * known_mst_weight, f'Proposed MST weight {my_mst_weight} is not within {allowed_error * 100}% of the networkx MST weight {known_mst_weight}'




