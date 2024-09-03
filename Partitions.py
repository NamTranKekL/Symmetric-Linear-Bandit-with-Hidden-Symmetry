import copy
import numpy as np
import random

def generate_list_of_lists(d):
    return [[i] for i in range(0, d)]

# Key: we must sorted every block in increasing order
def Refine_Partition_NC(Partition): #List # Given an edge, define new finest NC partition.
    Partition_New =  []
    for i in range(0, len(Partition)):
        Partition_New_element = copy.deepcopy(Partition)
        if len(Partition[i]) != 1:
            newblock = Refine_Block_NC(Partition[i]) # collection of new block
#            Partition_New_element.pop(i) # Remove the old block from the list
            for j in range(0, len(newblock)):
 #               Partition_New_element_add = copy.deepcopy(Partition_New_element) # keep a copy of the original after removing the old block
  #              Partition_New_element_add.append(newblock[j]) # add new block into the copy, may need repeat this step as there will be two list
   #             Partition_New.append(Partition_New_element_add)
                Partition_New_element_add = copy.deepcopy(Partition_New_element)
#                print(newblock[j])
                Partition_New_element_add[i:i+1] = newblock[j]
                Partition_New.append(sorted(Partition_New_element_add, key=lambda subset: subset[0]))

    return Partition_New

def Refine_Block_NC(block):
    # define new edge
    CollectionNewBlock = []
    if len(block) == 2:
        CollectionNewBlock = [ [[block[0]], [block[1]]]  ]
        return CollectionNewBlock
    else:
        # define edges
        for i in range(0, len(block)-1):  # before -1
            for j in range(i + 1, len(block)):
                edge = [i, j]                        # define edges
                newblock = Refine_Block_NC_withEdge(block, edge)
                CollectionNewBlock.append(newblock)
    return CollectionNewBlock

def Refine_Block_NC_withEdge(block, edge):
    start, end = edge[0], edge[1]
    # Partition the block into block_out and block_in, method using duality of NC lattice (circular representation).
    block_in = block[start:end]  # Inclusive slice
    block_out = block[:start] + block[end:]  # Elements outside the edge
    if block_out[0] < block_in[0]:
        return [block_out, block_in]
    else:
        return [block_in, block_out]


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



######################## Find dual of a NC partition - by myself ########################
def construct_dual_partition(partition, n):
    dual_partition = []
    dual_partition_relabel = relable_dual(partition, n)
    arcs_dual = collection_arcs_primal(dual_partition_relabel,n)
    dual_partition = arcs_to_partition(arcs_dual, n)
    return dual_partition

def collection_arcs_primal(partition, n): # Assuming partition is sorted with ascending order
    # test for partition = [[1,2,6],[3],[4,5]]
    arcs = []
    for i in range(0,len(partition)):
        for j in range(0,len(partition[i])-1):
            arcs.append([partition[i][j],partition[i][j+1]])
    return arcs

def relable_dual(partition, n):
    partition_dual_unsorted = []
    for i in range(0,len(partition)):
        block = [n-1 - x for x in reversed(partition[i])] # dual is in reverse order of primal
        partition_dual_unsorted.append(copy.deepcopy(block))

    partition_dual_sorted = sorted(partition_dual_unsorted, key=lambda subset: subset[0])
    return partition_dual_sorted

def arcs_to_partition(arcs_dual, n):
    full_set = list(range(0, n))
    block_in = []
    block_out = copy.deepcopy(full_set)
    for i in range(0, len(arcs_dual)):
        block_within_edge = list(range(arcs_dual[i][0] + 1, arcs_dual[i][1] + 1))
        block_in.append(block_within_edge)
        block_out = sorted(list(set(block_out) - set(block_within_edge)))

    partition_dual = [block_out] + block_in # need a bracket to block out, just a trick
    partition_dual = sorted(partition_dual, key=lambda subset: subset[0])

    ## Filtering out overlap elements
    for i in range(0,len(partition_dual)):
        for j in range(i+1,len(partition_dual)):
            partition_dual[i] = sorted(list(set(partition_dual[i]) - set(partition_dual[j])))
    return partition_dual

######################## Find neighbor of Coarsen_partition ########################

def Coarsen_Partition_NC(Partition, n): # For each edge.
    Partition_dual = construct_dual_partition(Partition, n)
    Partition_New = []
    for i in range(0, len(Partition_dual)):
        Partition_New_element = copy.deepcopy(Partition_dual)
        if len(Partition_dual[i]) != 1:
            newblock = Refine_Block_NC(Partition_dual[i])  # collection of new block
            #            Partition_New_element.pop(i) # Remove the old block from the list
            for j in range(0, len(newblock)):
                Partition_New_element_add = copy.deepcopy(Partition_New_element)
                Partition_New_element_add[i:i + 1] = newblock[j]
                Partition_New.append(construct_dual_partition(Partition_New_element_add, n)) # reverse to primal

    return Partition_New


def random_partition(d,k): # set d elements, starting from 0; k classes.
    Partition = [list(range(0, d))]
    for class_par in range(2,k+1):
        #valid_blocks = [block for block in Partition if len(block) >= 2]
        valid_blocks_with_indices = [(index, block) for index, block in enumerate(Partition) if len(block) >= 2]
        index_chosen, block_chosen = random.choice(valid_blocks_with_indices)
     #   block_chosen = random.choice(valid_blocks)
        edge_random = sorted(random.sample(list(range(0,len(block_chosen))),2))
        new_random_block = Refine_Block_NC_withEdge(block_chosen, edge_random)
        Partition[index_chosen:index_chosen+1] = new_random_block
    return sorted(Partition, key=lambda subset: subset[0])

def check_partition(partition_true, partition_est): # both list is sorted
    return partition_true == partition_est

#partition = [[1],[2],[3],[4],[5]]
#partition = [[1,2,3],[4,5]]
#partition = [[1,2,6],[3,4,5],[7]]
#partition = [[0],[1],[2],[3],[4],[5],[6]]
#n = 7

#print(construct_dual_partition(partition, n)) # for partition = [[1,2,3],[4,5]], code is wrong
#print(collection_arcs_primal(partition,n))
#print(relable_dual(partition,n))
#print(Refine_Partition_NC(partition))

#print(Refine_Block_NC_withEdge(block, edge))
#for i in range(0, len(block)):
 #   print(i)
#print(Refine_Block_NC(block))
#print(len(Refine_Block_NC(block)))

#print(Refine_Partition_NC(partition))
#print(len(Refine_Partition_NC(partition)))




#print(construct_dual_partition(partition, n))
#print(Coarsen_Partition_NC(partition, n))
#print(len(Coarsen_Partition_NC(partition, n)))
#partition = random_partition(10,3)
#print(partition)
#print(Refine_Partition_NC(partition))
#partition_est = copy.deepcopy(partition)
#partition_est[2] = random.shuffle(partition_est[2])
#print(check_partition(partition, partition_est))