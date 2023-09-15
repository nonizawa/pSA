import numpy as np
import time
import random
import math
import csv
import sys
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd

starttime = time.time()

# Annealing function (returns the spin array after annealing and the annealing time)
def annealing(tau, I0_min, I0_max, beta, nrnd, Mshot, J_matrix, spin_vector, Itanh_ini, async_prop):
    Itanh = Itanh_ini
    start_time = time.time()
    for i in range(Mshot): 
        I0 = I0_min
        while I0 <= I0_max:
            for i in range(tau):
                rnd = (1.0 + 1.0) * np.random.rand(vertex,1) - 1.0
                I_vector = np.dot(J_matrix, spin_vector)
                Itanh = np.tanh(I0*I_vector) + nrnd * rnd
                random_indices = np.random.choice(vertex, size=np.int32(vertex*async_prop), replace=False)
                spin_vector[random_indices] = np.where(Itanh[random_indices] >= 0, 1, -1)
                #print(spin_vector.shape)
            I0 = I0 / beta
    end_time = time.time() 
    annealing_time = end_time - start_time
    return (spin_vector, annealing_time)

# Function to calculate cut number
def cut_calculate(G_matrix, spin_vector):
    spin_vector_reshaped = np.reshape(spin_vector, (len(spin_vector),)) # spin_vectorを1次元配列に変換
    upper_triangle = np.triu_indices(len(spin_vector), k=1) # 上三角行列のインデックスを取得
    cut_val = np.sum(G_matrix[upper_triangle] * (1 - np.outer(spin_vector_reshaped, spin_vector_reshaped)[upper_triangle])) # 上三角行列の要素のみを計算してcut_valを算出
    return int(cut_val/2)

# Function to calculate energy
def energy_calculate(J_matrix, spin_vector):
        Jm_tmp = np.dot(J_matrix, spin_vector)
       # hm_tmp = np.dot(h_vector, spin_vector.T)
        return -np.sum(Jm_tmp * spin_vector) / 2 #- hm_tmp

# Create adjacency matrix for the graph
def get_graph(vertex, lines):
    G_matrix = np.zeros((vertex, vertex), int)
    
    # Counting the number of lines (edges)
    line_count = len(lines)
    print('Number of Edges :', line_count)
    
    # Iterating through the lines to construct the adjacency matrix
    for line_text in lines:
        weight_list = list(map(int, line_text.split(' ')))  # Convert space-separated string to list of integers
        i = weight_list[0] - 1
        j = weight_list[1] - 1
        G_matrix[i, j] = weight_list[2]  # Assign weight to the corresponding entry in the matrix
    
    # Adding the matrix to its transpose to make it symmetric
    G_matrix = G_matrix + G_matrix.T
    return G_matrix

# Read and process the file
def read_and_process_file(file_path):

    name = [None] * 5
    # Read the input file
    dir_path, name[0] = os.path.split(file_path)

    with open(file_path, 'r') as f:
        # Read lines from the file
        name[1] = f.readline().strip()  # number of nodes
        name[2] = f.readline().strip()  # edge value: bipolar or unipolar
        name[3] = f.readline().strip()  # edge type
        name[4] = f.readline().strip()  # best-known value
        lines = f.readlines()

    # Parse the vertex count from the file and create graph matrix
    vertex = np.int32(name[1])
    G_matrix = get_graph(vertex, lines)
    J_matrix = -G_matrix
    best_known = np.int32(name[4])

    return vertex, G_matrix, J_matrix, best_known, name 

# Create a command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--stall_prop', type=float, default=0.5, help='stalled prop値')
parser.add_argument('--graph_file', type=str, default='graph/G1.txt', help='グラフファイルのパス')
parser.add_argument('--cycle', type=int, default=1000, help="Number of cycles (default: 1000)")
parser.add_argument('--trial', type=int, default=100, help="Number of trials (default: 100)")
parser.add_argument('--tau', type=int, default=1, help="tau (default: 1)")
parser.add_argument('--gamma', type=float, default=0.1, help="gamma (default: 0.1)")
parser.add_argument('--delta', type=float, default=10, help="delta (default: 10)")
args = parser.parse_args()

# Retrieve values for stall_prop and graph_file
async_prop = np.float32(1-args.stall_prop)
vertex, G_matrix, J_matrix, best_known, name = read_and_process_file(args.graph_file)

# ------ Graph structure ------ #
mean_each = []
std_each = []
for j in range(vertex):
    mean_each.append((vertex-1)*np.mean(J_matrix[j]))
    std_each.append(np.sqrt((vertex-1)*np.var(J_matrix[j])))
sigma = np.mean(std_each)
mean = np.mean(mean_each)
print('mean = ',mean, 'sigma = ',sigma)

# ------ SpSA parameters ------ #
min_cycle = np.int32(args.cycle)
trial  = np.int32(args.trial)
Mshot = np.int32(1)
nrnd   = np.float32(1)
delta = np.float32(args.delta)
gamma = np.float32(args.gamma)
I0_min = np.float32(gamma/sigma)
I0_max = np.float32(delta/sigma)
tau    = np.int32(1)
beta   = np.float32((I0_min/I0_max)**(tau/(min_cycle-1)))
max_cycle = math.ceil((math.log10(I0_min/I0_max)/math.log10(beta))) * tau

# ------ Execution Program ------ #
print('trials:', trial)
print("Min Cycles :", min_cycle)
print('beta:', beta)
print('I0_min:', I0_min)
print('I0_max:', I0_max)
print('tau:', tau)
#print('nrnd vector', nrnd_vector)

Itanh_ini = (np.random.randint(0, 3, (vertex, 1))  - 1) * I0_min
cut_sum = 0
time_sum = 0
cut_list = []
for k in range(trial):
    ini_spin_vector = np.random.randint(0,2,(vertex, 1))    #0と1のランダムスピン配列初期値
    ini_spin_vector = np.where(ini_spin_vector == 0, -1, 1) #-1と1のスピン配列に変換
    (last_spin_vector, annealing_time) = annealing(tau, I0_min, I0_max, beta, nrnd, Mshot, J_matrix, ini_spin_vector, Itanh_ini, async_prop)
    cut_val = cut_calculate(G_matrix, last_spin_vector)
    min_energy = energy_calculate(J_matrix, last_spin_vector)
    cut_sum += cut_val
    cut_list.append(cut_val)
    time_sum += annealing_time
    print('File_name:',name[0], "Trial", k + 1, 'Cut value :', cut_val,'Min energy :', min_energy, 'Annealing time :',annealing_time)
cut_average = cut_sum / trial
cut_max = max(cut_list)
cut_min = min(cut_list)
time_average = time_sum / trial
print('Cut value average :', cut_average)
print('Cut value max :', cut_max)
print('Cut value min :', cut_min)
print('Time average :', time_average)

#df = pd.DataFrame(cut_list, columns=['cut_list'])
#df.to_csv('cut_list.csv', index=False)

print("Total time:", time.time()-starttime)

# Output file
data = [
    name[0],name[1],name[2],name[3],name[4], args.stall_prop, gamma, delta, cut_average, cut_max, cut_min, 100*cut_average/best_known, 100*cut_max/best_known, time_average]

if os.path.isfile("./result/result_SpSA.csv"): # "result.csv" ファイルが存在する場合
    with open("./result/result_SpSA.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
else: # "result_cpu.csv" ファイルが存在しない場合
    with open("./result/result_SpSA.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['graph','#node', 'Weights','Structure','Best-known value',	'stall_prop','gamma','delta','mean','max','min','mean%','max%','sim_time'])
        writer.writerow(data)


