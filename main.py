# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import copy
import pandas as pd
from sklearn.linear_model import Lasso
from tqdm import tqdm
import itertools

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


# Expected reward function to generate data  output = input_f (x,theta)
def expected_reward(x,theta):
    f = np.dot(x,theta)
    return f

# Generate non-crossing partition with number of classes d_0
def make_partitions(elements):
    yield from _make_partitions(sorted(elements, reverse=True), [], [])


def _make_partitions(elements, active_partitions, inactive_partitions):
    if not elements:
        yield active_partitions + inactive_partitions
        return

    elem = elements.pop()

    # Make create a new partition
    active_partitions.append([elem])
    yield from _make_partitions(elements, active_partitions, inactive_partitions)
    active_partitions.pop()

    # Add element to each existing partition in turn
    size = len(active_partitions)
    for part in active_partitions[::-1]:
        part.append(elem)
        yield from _make_partitions(elements, active_partitions, inactive_partitions)
        part.pop()

        # Remove partition that would create a cross if new elements were added
        inactive_partitions.append(active_partitions.pop())

    # Add back removed partitions
    for _ in range(size):
        active_partitions.append(inactive_partitions.pop())

    elements.append(elem)

def make_partitions_withclasses(elements, NumClass):
    partitionNumClass = []
    for partition in make_partitions(elements):
        if len(partition) == NumClass:
            partitionNumClass.append(copy.deepcopy(partition))
    return partitionNumClass
# Projection corresponding to a partition

# def projection_partition(theta, pi) pi is list,
def projection_matrix_partition(partition, n):
    P = np.zeros((n, n))

    for part in partition:
        size = len(part)
        for i in part:
            for j in part:
                P[i, j] = 1.0 / size
    return P

def matrix_Sparsity_IntervalPartition(d):
    W = np.zeros((d, d))
    for i in range(d):
        if i == 0:
            W[i, i] = 1
            W[i, i+1] = -1
        elif i == d-1:
            W[i, i] = 1
        else:
            W[i, i] = 1
            W[i, i+1] = -1
    return W

# Example usage:
#for partitions_ab in make_partitions_withclasses([0, 1, 2, 3], 2 ):
#partitions_ab = make_partitions_withclasses([0, 1, 2, 3], 2 )[1]
 #   print(projection_matrix_partition(copy.deepcopy(partitions_ab), 4))
# Search within the best partition.
# call n samples from the function
# Compute \hat \theta
# compute distance \hat \theta to any projection of m, see which one is smallest

d  = 16
d_0 = 2
#n = 20
Iter_T = np.linspace(20,1000,20)
Data_save = []
SampleEachT = 10
regret_data = {}
#collection_partitions = copy.deepcopy(make_partitions_withclasses(np.arange(d), d_0 ))
with tqdm(total=1) as pbar:

    for T in Iter_T:
        regret_data.update({"Regret_MS_" + str(T): np.zeros(SampleEachT), "Regret_Lasso_"+str(T): np.zeros(SampleEachT)})
        for insample in range(SampleEachT):
    #        print(T)
            theta = np.concatenate((1*np.ones(2), 2*np.ones(3), 1*np.ones(d-5)),axis=None)
    #        theta = np.concatenate((1*np.ones(2), 2*np.ones(d-2)),axis=None)
            n = round(d_0*pow(T,2/3))
            #n = d
            ## data generating and least square
            X = np.random.randn(n, d)
            Y = X @ theta + 0.1*np.random.randn(n)
            theta_est = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y
            #print(np.arange(d))

 #           collection_partitions = copy.deepcopy(make_partitions_withclasses(np.arange(d), d_0 ))

            ## need to rewrite the line the compute theta_proj
            min_ProjErr = 1000000
 #           partition_best = []
            theta_proj = []
 #           for partition_NC in collection_partitions:
  #              Proj = projection_matrix_partition(partition_NC,d)
   #             ProjErr = np.linalg.norm(theta_est - Proj @ theta_est )
    #            if ProjErr < min_ProjErr:
     #               min_ProjErr = ProjErr
      #              partition_best = copy.deepcopy(partition_NC)
                    # print(partition_best)
            #print(partition_best)



        #    theta_proj = projection_matrix_partition(partition_best,d) @ theta_est
         #   error_MS = np.linalg.norm( theta_proj - theta )
            error_LS = np.linalg.norm( theta_est - theta )
        #    print("Model Selection Risk Error = ", error_MS )
         #   print("Least Square Risk Error = ", error_LS )

            W_x_to_z = np.linalg.inv(matrix_Sparsity_IntervalPartition(d).T)
            Z = X @ W_x_to_z.T

            phi_Lasso = copy.deepcopy(Lasso(alpha = 0.05).fit(Z, Y).coef_) # alpha is the regularised constant, if s large, alpha should be large.
            theta_Lasso = W_x_to_z.T @ phi_Lasso
            error_Lasso = np.linalg.norm( theta_Lasso - theta )
        #    print("Lasso Risk Error = ", error_Lasso)

            #print(W_x_to_z)
         #   Regret_MS = n + error_MS*(T-n)
            Regret_Lasso = n + error_Lasso*(T-n)
    #        Regret_LS = n + error_LS*(T-n)
         #   regret_data["Regret_MS_" + str(T)][insample] = Regret_MS
            regret_data["Regret_Lasso_" + str(T)][insample] = Regret_Lasso
            #new_data = {"Regret_MS_" + str(T): Regret_MS, "Regret_Lasso_"+str(T): Regret_Lasso}
 #       Data_save.append(regret_data)
        pbar.update(1)

    df = pd.DataFrame(regret_data)
    excel_file_path = 'data1' + '.xlsx'

    df.to_excel(excel_file_path, index=False)

    print(f"Data saved to {excel_file_path}")
 #   print(regret_data['Regret_MS_20'])
