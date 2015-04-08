import csv
import numpy as np
from numpy import linalg as LA

def read_data():

	####
	# Read file once to determine number of nodes instead of hard coding it
	####

	adjacency_matrix = np.zeros((4039,4039))
	degree_matrix = np.zeros((4039,4039))
	
	# Build adjacency matrix edge by edge
	with open('facebook_combined.txt', 'rU') as fp:

		for line in fp:
			edge = line.split(' ')
			node1 = int(edge[0])
			node2 = int(edge[1])

			# Set adjacency matrix values
			adjacency_matrix[node1,node2] = 1
			adjacency_matrix[node2,node1] = 1

			# Set degree matrix values
			degree_matrix[node1,node1] += 1
			degree_matrix[node2,node2] += 1
		
	fp.close()

	laplacian_matrix = np.subtract(degree_matrix,adjacency_matrix)

	return laplacian_matrix

if __name__=='__main__':

	# Get Laplacian matrix representing graoh network from data file containing edges
	laplacian_matrix = read_data()

	# Get eigenvalues and eigenvectors of Laplacian matrix
	eigenvalues, eigenvectors = LA.eig(laplacian_matrix)

	# Naive clustering of graph into 2 communities
	community_1 = 0
	community_2 = 0
	cluster_array = []
	for i,ele in enumerate(eigenvectors[:,1]):
		if ele >= 0:
			community_1 += 1
			cluster_array.append(1)
		else:
			community_2 += 1
			cluster_array.append(2)

	print community_1
	print community_2
	print len(cluster_array)

	# Extension: Vary k (number of communities); cluster points of graph into k clusters