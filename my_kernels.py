import numpy as np
from sklearn.metrics import mean_squared_error, pairwise
import networkx as nx
from node2vec import Node2Vec




def create_laplacian_matrix(edge_list, n_nodes):
	adj_matrix = np.zeros((n_nodes, n_nodes))
	for u, f in edge_list:
		adj_matrix[u,f] = 1 
	L = np.diag(np.sum(adj_matrix, axis = 1)) - adj_matrix

	return L


def inv_commute_time_kernel(edge_list, n_nodes):
	return create_laplacian_matrix(edge_list,n_nodes)

def inv_regularized_laplacian_kernel(edge_list,n_nodes):

	L = create_laplacian_matrix(edge_list,n_nodes)
	gamma = 0.1 

	return 1 + gamma * L 

def inv_diffusion_kernel(edge_list, n_nodes):
	L = create_laplacian_matrix(edge_list,n_nodes)
	beta = 0.1
	return np.linalg.pinv(np.exp(-beta * L))


def inv_node2vec_kernel(edge_list, n_nodes):
	print ("computing node2vec kernel matrix")
	G = nx.Graph()
	G.add_edges_from(edge_list)
	dimensions = 32
	node2vec = Node2Vec(G, dimensions = dimensions, walk_length = 20, num_walks = 100)
	# Embed nodes
	model = node2vec.fit(window=10, min_count=1, batch_words=4)
	print ("generating embedding matrix")
	embed_matrix = np.zeros((n_nodes, dimensions))
	for i in range(n_nodes):
		if str(i) in model.wv: 
			embed_matrix[i] = model.wv[str(i)]
	return inv_rbf_kernel(embed_matrix)



def inv_rbf_kernel(feature_matrix):
	return np.linalg.inv(pairwise.rbf_kernel(feature_matrix))




def inv_polynomial_kernel(feature_matrix):
	return np.linalg.inv(pairwise.polynomial_kernel(feature_matrix))