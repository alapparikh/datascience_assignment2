import csv
import numpy as np

def read_data():

	adjacency_matrix = np.zeros((4039,4039))
	
	# Build adjacency matrix edge by edge
	with open('facebook_combined.txt', 'rU') as fp:

		for line in fp:
			edge = line.split(' ')
			adjacency_matrix[int(edge[0]),int(edge[1])] = 1.
			adjacency_matrix[int(edge[1]),int(edge[0])] = 1.
		
	fp.close()

	print adjacency_matrix

if __name__=='__main__':

	read_data()