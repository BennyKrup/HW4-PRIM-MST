import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None
        #prims algorithm
        num_vertices = len(self.adj_mat)
        mst = np.zeros_like(self.adj_mat)
        
        # List to keep track of visited vertices. Start with vertex 0 as visited.
        visited = [False] * num_vertices
        visited[0] = True

        # Priority queue for edges (weight, vertex1, vertex2)
        edges = [(self.adj_mat[i][0], 0, i) for i in range(1, num_vertices) if self.adj_mat[i][0] > 0]
        heapq.heapify(edges)  # Transform list into a heap

        while edges:
            weight, from_vertex, to_vertex = heapq.heappop(edges)  # Pop the edge with the smallest weight

            if not visited[to_vertex]:
                # Add edge to MST
                mst[from_vertex][to_vertex] = weight
                mst[to_vertex][from_vertex] = weight  # Since the graph is undirected

                visited[to_vertex] = True  # Mark vertex as visited

                # Add new edges from the newly visited vertex to the queue
                for next_vertex, edge_weight in enumerate(self.adj_mat[to_vertex]):
                    if edge_weight > 0 and not visited[next_vertex]:
                        heapq.heappush(edges, (edge_weight, to_vertex, next_vertex))

        self.mst = mst
