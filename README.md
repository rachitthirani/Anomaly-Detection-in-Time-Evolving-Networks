# Project 4: Anomaly Detection in Time Evolving Networks
Project members:
1. Manjusha Trilochan Awasthi (mawasth)
2. Kriti Shrivastava (kshriva)
3. Rachit Thirani (rthiran)

This is an implementation for the algorithm from the paper 4: NetSimile: A Scalable Approach to Size-Independent Network Similarity

Python version used: Python 3.6.0

OS used: Windows 10 64bit (RAM:16GB)

Python libraries needed:

1. Networkx: install using the command "pip install networkx" or follow the instructions mentioned here https://networkx.github.io/documentation/development/install.html
2. Numpy: install using the command "pip install numpy" or follow the intructions mentioned here https://docs.scipy.org/doc/numpy/user/install.html

Instructions to run the program:
1. Go the the directory containing the python script.
2. Give the command "python anomaly.py graphname" where graphname is the name of the graph you want to find the communities for.
   ex. python anomaly.py voices

Output of the program:
1. Text files: Contains the similarity scores (Canberra distance) between the consequent graphs in the time series. These files are generated in the directory containing the python script under /output/files/. Ex."voices_time_series.txt"
2. Graph plots: Plot between the similarity score (Canberra distance) between the graphs and the time series. The red horizontal line is the threshold used for detecting anomalies. These plots are generated in the directory containing the python script under /output/plot/. Ex."voices_time_series.png" 
