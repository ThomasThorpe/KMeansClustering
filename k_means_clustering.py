import csv
import math
import random
import os

import numpy as np

# k is the number of clusters that data can be assigned to.
# cluster_names is a list of names for the clusters.
# input_file is a csv file to be parsed into a numpy array.
# output_file is the file to write results to.
class KMeansCluster:
    def __init__(self,k,cluster_names,input_file,output_file):
        # Validate input. Assuming types are correct.
        if(len(cluster_names) != k):
            raise IndexError("There must be exactly as many cluster names as there are clusters!")
        if not(os.path.isfile("./" + input_file)):
            raise IOError("There was an error finding the input file!")

        # Init members.
        self._k = k
        self._name_map = self._map_names(cluster_names)
        self._input_file = input_file
        self._output_file = output_file
        self._dimensions = self._set_dimensions()
        self._num_rows = self._set_num_rows()
        self._centroids = np.zeros((self._k,self._dimensions),dtype=np.float64)
        self._init_centroids()
        self._input_data = np.zeros((self._num_rows,self._dimensions),dtype=np.float64)
        self._current_cluster = np.zeros((self._num_rows,1),dtype=np.int16)
        self._read_input()  # Sets up input data into numpy array.
        self._total_error_metric = None
        return

    def em_algorithm(self):
        lastErrorMetric = 0
        interation = 0
        while(self._total_error_metric != lastErrorMetric):
            if interation != 0:
                lastErrorMetric = self._total_error_metric
            self._expectation_step()
            self._maximisation_step()
            interation += 1
            print("{0} | {1}".format(self._total_error_metric,lastErrorMetric))
        self._write_results()
        return

    def _expectation_step(self):
        # Each data point is assigned to closest cluster centroid measured using Euclidian distance metric.
        self._total_error_metric = 0
        for row in range(self._num_rows):
            closestDistance = float("inf")
            closestCentroid = None
            for centroid in range(len(self._centroids)):
                tmp = 0
                for dimension in range(self._dimensions):
                    tmp += (self._input_data[row][dimension]-self._centroids[centroid][dimension])**2
                tmp = math.sqrt(tmp)
                if tmp < closestDistance or closestDistance is None:
                    closestDistance = tmp
                    closestCentroid = centroid
            self._current_cluster[row][0] = closestCentroid
            self._total_error_metric += closestDistance
        return
    
    def _maximisation_step(self):
        # Minimise total_error_metric by adjusting the cluster centroids to the centre of data points assigned to it.
        totals = np.zeros((self._dimensions,len(self._centroids)),dtype=np.float64)  # Store totals for each dimension for each centroid.
        counts = np.zeros(len(self._centroids))  # Count how many data points assigned to each centroid.
        
        for row in range(self._num_rows):
            counts[self._current_cluster[row][0]] += 1
            for dimension in range(self._dimensions):
                totals[dimension][self._current_cluster[row][0]] += self._input_data[row][dimension]
        # Update centroids
        for centroid in range(len(self._centroids)):
            for dimension in range(self._dimensions):
                self._centroids[centroid][dimension] = totals[dimension][centroid] / counts[centroid]
        return

    def _set_dimensions(self):
        d = None
        with open(self._input_file,'r',newline='\n') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            row = reader.__next__()
            d = len(row)
        return d

    def _set_num_rows(self):
        n = None
        with open(self._input_file,'r',newline='\n') as csvfile:
            n = sum(1 for line in csvfile)
        return n

    def _init_centroids(self):
        for centroid in range(self._k):
            for dimension in range(self._dimensions):
                self._centroids[centroid][dimension] = random.uniform(-10,10)
        return

    def _map_names(self,cluster_names):
        nameMap = {}
        for i in range(self._k):
            nameMap[i] = cluster_names[i]
        return nameMap

    def _read_input(self):
        # Read each row and parse into numpy array of input data.
        lineNum = 0
        with open(self._input_file,'r',newline='\n') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
                for i in range(self._dimensions):
                    self._input_data[lineNum][i] = row[i]
                lineNum += 1
        return
            
    def _write_results(self):
        with open(self._output_file,'w',newline='\n') as f:
            f.write("error = {0:.3f}\n".format(self._total_error_metric))
            for row in range(self._num_rows):
                f.write(self._name_map.get(self._current_cluster[row][0])+'\n')
        return


if __name__ == "__main__":
    k = 5
    centroids = np.array([[-0.357,-0.253,-0.1],[-0.055,4.392,0.1],[2.674,-0.001,0.2],[1.044,-1.251,-0.2],[-1.495,-0.090,0.3]],np.float64)
    input_file = "data.csv"
    output_file = "OUTPUT.TXT"
    cluster_names = ["Adam","Bob","Charley","David","Edward"]
    try:
        my_cluster = KMeansCluster(k,cluster_names,input_file,output_file)
    except (IndexError, IOError) as e:
        print(e)
    else:
        my_cluster.em_algorithm()