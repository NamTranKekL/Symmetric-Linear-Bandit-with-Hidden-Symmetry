# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import copy
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
import cvxpy as cp

from tqdm import tqdm
import itertools
from Partitions import projection_matrix_partition, matrix_Sparsity_IntervalPartition, generate_list_of_lists, Coarsen_Partition_NC, check_partition,random_partition
from PopArt import objective_fn

from my_catoni import *


d = 80
d0 = 10
Iter_T = np.linspace(400,400,1)
SampleEachT = 100

#theta = np.concatenate((1*np.ones(2), 2*np.ones(3), 1*np.ones(d-5)),axis=None)
#partition_true = [[1, 5, 6, 7, 8, 9], [0], [2, 3, 4]]
#partition_true = [[1, 2, 3, 4, 5, 6], [0], [7, 8, 9]]
#partition_true = [[39], [0, 1, 2, 3, 4, 5, 6, 31, 32, 33, 34, 35, 36, 37, 38], [7, 8, 9, 10, 11, 12, 16, 17, 18, 24, 28, 29, 30], [19, 21, 23], [22], [13, 14, 15], [25, 26, 27], [20]]
#partition_true = [[1, 2, 3, 4, 5, 6, 7, 8, 89, 90, 91, 95, 96, 97, 98, 99], [9, 10, 11, 12, 13, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88], [53], [69], [0], [14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [38, 39], [48, 49, 50], [92, 93, 94], [66, 67, 68]]
#partition_true = [[0,99],[1,98],[2,97],[3,96],[4,95],[5,94],[6,93],[7,92],[8,91],list(range(9,91))]
#partition_true = [[0,19],[1,18],list(range(2,18))]
#partition_true = [[0,39],[1,38],[2,37],[3,36],list(range(4,36))]
#theta_raw = 10*np.random.randn(d)
#theta = np.random.randn(d)
#theta = projection_matrix_partition(partition_true, d) @ theta_raw

regret_data = {}

with tqdm(total=1) as pbar:
    for T in Iter_T:
 #       regret_data.update({"Regret_Lasso_" + str(T): np.zeros(SampleEachT), "Regret_PopArt_"+str(T): np.zeros(SampleEachT)})
        regret_data.update({"Regret_Lasso_" + str(T): np.zeros(SampleEachT)})
        for insample in range(SampleEachT):
            partition_true = random_partition(d,d0)
            theta_raw = 10 * np.random.randn(d)
            theta = projection_matrix_partition(partition_true, d) @ theta_raw

            n = round(pow(d0,1/3)*pow(T,2/3))
 #           print(n)
            #X = normalize(np.random.randn(n, d), axis=1, norm='l2')
            X = np.random.randn(n, d)
            Y = X @ theta + 0.1*np.random.randn(n)
            theta_est = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y
            ##################################################################################################################
            partition = generate_list_of_lists(d)
            partition_best = partition


            W_x_to_z = np.linalg.inv(matrix_Sparsity_IntervalPartition(d).T)
            Z = X @ W_x_to_z.T

        ########## Lasso experiment ##################################################
#            phi_Lasso = copy.deepcopy(Lasso(alpha=4 * 0.1*np.sqrt(np.log(d)/ n)).fit(Z, Y).coef_)  # alpha is the regularised constant, if s large, alpha should be large.

            beta = cp.Variable(d)
            lambd_b = 8* 0.1 * np.sqrt(np.log(d) * n)
            lassosol = cp.Problem(cp.Minimize(objective_fn(Z, Y, beta, lambd_b)))
            lassosol.solve()
            phi_Lasso = beta.value

            theta_Lasso = W_x_to_z.T @ phi_Lasso
            error_Lasso = np.linalg.norm(theta_Lasso - theta)
   #         print(error_Lasso)
            Regret_Lasso = n + error_Lasso * (T - n)
            #        Regret_LS = n + error_LS*(T-n)
            regret_data["Regret_Lasso_" + str(T)][insample] = Regret_Lasso

            ########## PopArt experiment ##################################################
 #           M2 = 1
  #          vari = (d0 ** 2 + 0.1 ** 2) * M2  ### M2 = Cmin^{-1} = 1, Q_inv = identity matrix in this case
   #         #vari = 390
      #      threshold = width_catoni(n, d, 0.05, vari)
       #     X_hist=np.zeros((n,d))
        #    for t in range(0, n):
         #       X_hist[t] =Y[t] * Z[t,:]
#            phi_hat_raw = catoni_esti(X_hist, 0.05, vari)
          #  phi_hat_raw = np.mean(X_hist, axis=0)
           # phi_PopArt = (np.abs(phi_hat_raw) > threshold) * phi_hat_raw
            #theta_PopArt = W_x_to_z.T @ phi_PopArt

          #  error_PopArt = np.linalg.norm(theta_PopArt - theta)
           # Regret_PopArt = n + error_PopArt * (T - n)
            #regret_data["Regret_PopArt_" + str(T)][insample] = Regret_PopArt

        pbar.update(1)

    df = pd.DataFrame(regret_data)
    excel_file_path = 'data_Lasso_NC_'+str(d) + '.xlsx'

    df.to_excel(excel_file_path, index=False)

#        print(f"Data saved to {excel_file_path}")
## Clean the code, and run compare to lasso