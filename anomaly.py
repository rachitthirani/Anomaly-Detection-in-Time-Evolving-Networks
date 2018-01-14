# Implementation for paper 4: NetSimile: A Scalable Approach to Size-Independent Network Similarity

# Project Team Members:
# 1. Manjusha Trilochan Awasthi (mawasth)
# 2. Kriti Shrivastava (kshriva)
# 3. Rachit Thirani (rthiran)

import networkx as nx
import numpy as np
import os
import sys
import scipy
from scipy.stats import kurtosis, skew
import scipy.spatial.distance
from os.path import join
from pylab import *


def readGraph(graphName):
    # Input: name of the graph for which the files have to be read
    # Output: creates and returns a list of networkx graphs
    filePath = os.getcwd() + '/datasets/' + graphName
    # Read all files in the directory and create a networkx graph for each file
    graphList = list()
    for file in os.listdir(filePath):
        file = filePath + '/' + file
        f = open(file, 'r')
        # Skipping the first row containing number of nodes and edges
        next(f)
        g = nx.Graph()
        for line in f:
            line = line.split()
            g.add_edge(int(line[0]), int(line[1]))
        # Append the graph into the list
        graphList.append(g)
    return graphList


def calculateThreshold(dists):
    # Input: List of Canberra distances between graphs
    # Output: Upper threshold value for detecting anomalies
    # Caculating the upper threshold using the formula given in paper
    n = len(dists)
    sumMi = 0
    for i in range(2, n):
        Mi = abs(dists[i] - dists[i - 1])
        sumMi = sumMi + Mi
    M = sumMi / (n - 1)
    median = np.median(dists)
    threshold = median + 3 * M
    return threshold


def findAnomalies(dists, threshold):
    # Input: canberra distance between i and i+1 graphs
    # Output: index of the graphs which are the anomalies in the list of graphs
    anomalies = []
    for i in range(0, len(dists) - 1):
        # Graph is anomalous if there are two consecutive anomalous time points in the output
        if dists[i] >= threshold and dists[i + 1] >= threshold:
            anomalies.append(i)
    return anomalies


def generateTimeSeriesFile(dists):
    #Input: List of canberra distances between graphs
    #Output: Creates a text file with similarity values of the time series
    timeSeriesFile = "output/files/" + sys.argv[1] + "._time_series.txt"
    os.makedirs(os.path.dirname(timeSeriesFile), exist_ok=True)
    f = open(timeSeriesFile, 'w+')
    for dist in dists:
        f.write(str(dist) + '\n')
    f.close()


def plotTimeSeries(dists, threshold, anomalies):
    #Input: List of canberra distances between graphs and the upper threshold value for detecting anomalies
    #Output: Generates the time series plot for detecting anomalies indicating the threshold
    figure(figsize=(12, 6))
    plt.plot(dists, "-o")
    axhline(y=threshold, c='red', lw=2)
    plt.title("Anomaly Detection for " + sys.argv[1] + " dataset")
    plt.xlabel("Time Series")
    plt.ylabel("Canberra Distance")
    plotFile = "output/plot/" + sys.argv[1] + "_time_series.png"
    os.makedirs(os.path.dirname(plotFile), exist_ok=True)
    savefig(plotFile, bbox_inches='tight')


# Algorithm 1: NETSIMILE
def netSimile(graphList, doClustering):
    # Input: a list of graphs for which anomaly has to be detected and variable stating whether clustering has to be performed
    # Create node-feature matrices for all the graphs
    nodeFeatureMatrices = getFeatures(graphList)
    # generate 'signature' vectors for each graph
    signatureVectorList = aggregator(nodeFeatureMatrices)
    # do comparison and return similarity/distance values for the given graphs
    dists = compare(signatureVectorList, doClustering)
    # calculate upper threshold which will be used to determine the anomalies
    threshold = calculateThreshold(dists)
    # Find anomalous graphs on the basis of threshold
    anomalies = findAnomalies(dists, threshold)
    # Generate the time series text file
    generateTimeSeriesFile(dists)
    # Plot the time series indicating the threshold
    plotTimeSeries(dists, threshold, anomalies)


# Algorithm 2: NETSIMILE's GETFEATURES
def getFeatures(graphList):
    # Input: list of graphs for which feature list has to be generated
    # Output: a list of node*feature matrix for all the graphs in the graph list
    nodeFeatureMatrices = []
    # Compute the node*feature matrix for graph G
    for G in graphList:
        nodeFeatureMatrix = []
        # Calculated the values of the 7 features for all nodes in the graph G
        for node in G.nodes():
            # Feature 1: degree of the node
            d_i = G.degree(node)
            # Feature 2: clustering coefficient of the node
            c_i = nx.clustering(G, node)
            # Feature 3: average number of node's two-hop away neighbors
            d_ni = 0
            for neighbor in G.neighbors(node):
                d_ni = d_ni + G.degree(neighbor)
            d_ni = float(d_ni) / float(d_i)
            # Feature 4: average clustering coefficient of neighbors of the node
            c_ni = 0
            for neighbor in G.neighbors(node):
                c_ni = c_ni + nx.clustering(G, neighbor)
            c_ni = float(c_ni) / float(d_i)
            # Feature 5: number of edges in node's egonet
            egonet = nx.ego_graph(G, node)
            E_ego = len(egonet.edges())
            # Feature 6: number of outgoing edges from node's egonet
            Estar_ego = 0
            e_list = set()
            for vertex in egonet:
                # Finding all edges of the nodes in the egonet
                e_list = e_list.union(G.edges(vertex))
            # Removing the edges which are the part of egonet itself to get the outing edges
            e_list = e_list - set(egonet.edges())
            Estar_ego = len(list(e_list))
            # Feature 7: number of neighbors of egonet
            N_ego = 0
            n_list = set()
            for vertex in egonet:
                # Finding all neighbors of the nodes in the egonet
                n_list = n_list.union(G.neighbors(vertex))
            # Removing the nodes which are the part of egonet itself to get the remaining neighbors
            n_list = n_list - set(egonet.nodes())
            N_ego = len(list(n_list))
            nodeFeatureMatrix.append([d_i, c_i, d_ni, c_ni, E_ego, Estar_ego, N_ego])
        # Append the node*feature matrix for the graph to the list
        nodeFeatureMatrices.append(nodeFeatureMatrix)
    return nodeFeatureMatrices


# Algorithm 3: NETSIMILE's AGGREGATOR
def aggregator(nodeFeatureMatrices):
    # Input: a list of node*feature matrix for all the graphs in the list
    # Output: a list of signature vectors for all the graphs in the list
    signatureVectorList = list()
    for nodeFeatureMatrix in nodeFeatureMatrices:
        signatureVector = []
        # Calculate the aggregate values for all the 7 features
        for i in range(7):
            featureColumn = [node[i] for node in nodeFeatureMatrix]
            aggFeature = [np.median(featureColumn), np.mean(featureColumn), np.std(featureColumn),
                          skew(featureColumn), kurtosis(featureColumn, fisher=False)]
            # Append the aggregated values for this feature to the signature vector
            signatureVector = signatureVector + aggFeature
        signatureVectorList.append(signatureVector)
    return (signatureVectorList)


# Algorithm 4: NETSIMILE's COMPARE
def compare(signatureVectorList, doClustering):
    # Since clustering is out of scope of this project
    if (doClustering == False):
        # Calculate canberra distance between i and i+1 graphs 
        dist = [scipy.spatial.distance.canberra(signatureVectorList[i], signatureVectorList[i - 1])
                for i in range(1, len(signatureVectorList))]
    return dist


if __name__ == "__main__":
    # Read input file name and create file path accordingly
    graphName = sys.argv[1]
    # Read files and create a graph list
    graphList = readGraph(graphName)
    # Setting doClustering as false as clustering is out of scope of this project
    doClustering = False
    # Algoritm 1: NetSimile
    netSimile(graphList, doClustering)
