import csv
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from igraph import *

def read_data():

	####
	# TODO: Read file once to determine number of nodes instead of hard coding it
	####

	adjacency_matrix = np.zeros((4039,4039))
	degree_matrix = np.zeros((4039,4039))
	g = Graph()
	g.add_vertices(4039)
	
	# Build adjacency matrix edge by edge
	with open('facebook_combined.txt', 'rU') as fp:

		for line in fp:
			edge = line.split(' ')
			node1 = int(edge[0])
			node2 = int(edge[1])

			# Set iGraph
			g.add_edge(node1,node2)

			# Set adjacency matrix values
			adjacency_matrix[node1,node2] = 1
			adjacency_matrix[node2,node1] = 1

			# Set degree matrix values
			degree_matrix[node1,node1] += 1
			degree_matrix[node2,node2] += 1
		
	fp.close()

	laplacian_matrix = np.subtract(degree_matrix,adjacency_matrix)

	return laplacian_matrix, g

def get_cut_index (array):

	max_ratio = 0
	index = 0
	#print 'array 0: ', array[0]
	for i in range(1,len(array) - 1):
		#print array[i]
		if (array[i] - array[i-1] != 0):
			ratio = (array[i+1] - array[i])/(array[i] - array[i-1])
			if ratio > max_ratio:
				max_ratio = ratio
				index = i

	return index

def naive_cluster (eigenvector):

	community_1 = 0
	community_2 = 0
	cluster_array = []
	for i,ele in enumerate(eigenvector):
		if ele >= 0:
			community_1 += 1
			cluster_array.append(1)
		else:
			community_2 += 1
			cluster_array.append(2)

	print community_1
	print community_2
	print len(cluster_array)

	return cluster_array

if __name__=='__main__':

	nclusters = 8

	# Get Laplacian matrix representing graph network from data file containing edges
	laplacian_matrix, graph = read_data()

	# Get eigenvalues and eigenvectors of Laplacian matrix
	eigenvalues, eigenvectors = LA.eig(laplacian_matrix)

	# Sort eigenvalues and eigenvectors from smallest to largest
	idx = eigenvalues.argsort()   
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]

	# Naive clustering of graph into 2 communities based on 2nd eigenvalue
	#clustered_array = naive_cluster(eigenvectors[:,1])

	# Extension: Vary k (number of communities); cluster points of graph into k clusters using k-means clustering
	# Get index of max difference between 2 eigenvalues, this gives the number of communities to be formed
	nclusters = get_cut_index(eigenvalues)

	sliced_array = eigenvectors[:,:nclusters]
	#print sliced_array.shape
	#print eigenvectors.shape

	# K Means clustering using scikit-learn
	kmeans = KMeans(n_clusters=nclusters)
	results = kmeans.fit_predict(sliced_array)
	graph.vs['community'] = results

	# Calculate how many nodes in each cluster
	total_number = {}
	for result in results:
		if result in total_number:
			total_number[result] += 1
		else:
			total_number[result] = 1

	# Visualize results using iGraph
	color_dict = {0: "blue", 1: "green", 2: "red", 3: "yellow", 4: "pink", 5:"black", 6: "orange", 7: "brown"}
	layout = graph.layout('kk')

	# Graph style
	visual_style = {}
	visual_style['layout'] = layout
	visual_style['vertex_size'] = 3
	visual_style['vertex_color'] = [color_dict[community] for community in graph.vs['community']]
	plot(graph, **visual_style)
