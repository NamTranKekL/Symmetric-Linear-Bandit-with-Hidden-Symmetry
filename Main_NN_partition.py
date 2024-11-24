# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import copy
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize

from tqdm import tqdm
import itertools
from Partitions import projection_matrix_partition, matrix_Sparsity_IntervalPartition, generate_list_of_lists, check_partition,random_partition, Coarsen_Partition_NN, Hard_Partition_NN

import cvxpy as cp

from PopArt import objective_fn

#from my_catoni import *

d = 40
d0 = 4
Iter_T = np.linspace(400,20000,20)
SampleEachT = 20

regret_data = {}

with tqdm(total=1) as pbar:
    for T in Iter_T:
        regret_data.update({"Regret_MS_" + str(T): np.zeros(SampleEachT), "Regret_Lasso_"+str(T): np.zeros(SampleEachT)})
        for insample in range(SampleEachT):
#            print(insample)
            partition_true = Hard_Partition_NN(d,d0)
            theta_raw = 10 * np.random.randn(d)
            theta = projection_matrix_partition(partition_true, d) @ theta_raw
#            print("phi_sparse = ", matrix_Sparsity_IntervalPartition(d)@theta)

            n = round(pow(d0,1/3)*pow(T,2/3))
 #           print(n)
            X = np.sqrt(d)*normalize(np.random.randn(n, d), axis=1, norm='l2')
            #X = np.random.randn(n, d)
            Y = X @ theta + 0.1*np.random.randn(n)
            theta_est = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y
            ##################################################################################################################
            partition = generate_list_of_lists(d)
            partition_best = partition

            Proj = projection_matrix_partition(partition, d)
            PredictionError = np.linalg.norm(Y - X @ Proj @ theta_est)
            PredictionError_best = 10000000

            ##### Model selection
            for i in range(1,d-d0+1):
                PredictionError_best = 10000000
#                print(i)
                CollectionNeighbor = Coarsen_Partition_NN(partition_best,d)
            #    print(CollectionNeighbor)
                for j in range(len(CollectionNeighbor)):
                    Proj = projection_matrix_partition(CollectionNeighbor[j], d)
                    PredictionError = np.linalg.norm(Y - X @ Proj @ theta_est)
                    if PredictionError < PredictionError_best:
                        PredictionError_best =  PredictionError
                        partition_best = CollectionNeighbor[j]
            theta_ms = projection_matrix_partition(partition_best,d) @ theta_est
#            print("partition_est:",partition_best)
#            print("partition_true:",partition_true)
 #           print(check_partition(partition_true, partition_best))
            error_MS = np.linalg.norm( theta_ms - theta )
            error_LS = np.linalg.norm(theta_est - theta)
 #           print("error MS:",error_MS)
 #           print("error LS:",error_LS)

            ##### Lasso
            W_x_to_z = np.linalg.inv(matrix_Sparsity_IntervalPartition(d).T)
            Z = X @ W_x_to_z.T

#            phi_Lasso = copy.deepcopy(Lasso(alpha=4 * 0.1*np.sqrt(np.log(d)/ n)).fit(Z, Y).coef_)  # alpha is the regularised constant, if s large, alpha should be large.
            beta = cp.Variable(d)
            lambd_b = 8 * 0.1 * np.sqrt(np.log(d) * n)
            lassosol = cp.Problem(cp.Minimize(objective_fn(Z, Y, beta, lambd_b)))
            lassosol.solve()
            phi_Lasso = beta.value

            theta_Lasso = W_x_to_z.T @ phi_Lasso
            error_Lasso = np.linalg.norm(theta_Lasso - theta)
 #           print("Lasso Risk Error = ", error_Lasso)

            ##### Compute regret of both
            Regret_MS = n + error_MS*(T-n)
            Regret_Lasso = n + error_Lasso * (T - n)
            #        Regret_LS = n + error_LS*(T-n)
            regret_data["Regret_MS_" + str(T)][insample] = Regret_MS
            regret_data["Regret_Lasso_" + str(T)][insample] = Regret_Lasso

        pbar.update(1)

    df = pd.DataFrame(regret_data)
    excel_file_path = 'data_greedy_NN_' +str(d)+ '.xlsx'

    df.to_excel(excel_file_path, index=False)

#        print(f"Data saved to {excel_file_path}")
## Clean the code, and run compare to lasso
