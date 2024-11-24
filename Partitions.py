import copy
import numpy as np
import random
from itertools import combinations


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


def random_partition_NC(d,k): # set d elements, starting from 0; k classes.
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




################################################## Interval partition ##################################################
def Coarsen_Partition_Interval(Partition, n):
    Partition_New = []
    if len(Partition) == 1:
        Partition_New = [Partition]
    elif len(Partition) == 2:
        Partition_New = [[Partition[0] + Partition[1]]]
    else:
        for i in range(len(Partition)-1):
            New_Partition = Partition[:i] + [Partition[i] + Partition[i + 1]] + Partition[i + 2:]
            Partition_New.append(New_Partition)
    return Partition_New

def random_interval_partition(d,k):
    Partition = generate_list_of_lists(d)
    for i in range(1,d-k+1): # counter
        random_index = random.randint(0, len(Partition) - 2)
        Partition = Partition[:random_index] + [Partition[random_index] + Partition[random_index + 1]] + Partition[random_index + 2:]
    return Partition
    # start from  d = {{1},{2}....}
    # randomly choose block to merge


################################################## Nonnesting partition ##################################################


def create_edges(sorted_list):
    edges = []
    for i in range(len(sorted_list) - 1):
        # Create an edge between the current element and the next element
        edges.append([sorted_list[i], sorted_list[i + 1]])
    return edges

def is_nested(edge1, edge2):
    # Check if edge1 is nested within edge2
    return (edge2[0] < edge1[0] < edge1[1] < edge2[1]) or \
           (edge1[0] < edge2[0] < edge2[1] < edge1[1])

def check_nested_edges(list_A, list_B): # true if there is nested, false of non_nested
    for edge_A in list_A:
        for edge_B in list_B:
            if is_nested(edge_A, edge_B):
 #               print(edge_A, edge_B)
  #              print(is_nested(edge_A, edge_B))
                return True
    return False

def Coarsen_Partition_NN(Partition, n):
    # Get all combinations of two blocks from the partition
    Partition_New = []
    for (i, j) in combinations(range(len(Partition)), 2):
        # Merge the two selected blocks
        merged_block = sorted(Partition[i] + Partition[j])

        left_blocks = []
        # List the blocks that were not merged
        left_blocks = [Partition[k] for k in range(len(Partition)) if k != i and k != j]

        for Partition_left in left_blocks: # need to check all block and left_block
            if check_nested_edges(create_edges(merged_block),create_edges(Partition_left)): # there is nesting arcs, skip (i,j)
                break
        else:
            New_nonnesting_Partition = copy.deepcopy(left_blocks + [merged_block])
            Partition_New.append(copy.deepcopy(sorted(New_nonnesting_Partition, key=lambda subset: subset[0])))

    return Partition_New



def Random_Partition_NN(d, k): # need to fix
    # Get all combinations of two blocks from the partition
    partition = generate_list_of_lists(d)

    for counter in range(1, d - k + 1):  # counter
        print(counter)
        indices = list(range(len(partition)))
        tried_pairs = []
        NoNonestingCreated = True

        possible_merge = list(combinations(range(len(partition)), 2))

        while NoNonestingCreated:
 #           print(len(possible_merge))
            i, j = random.choice(possible_merge)

            merged_block = sorted(partition[i] + partition[j])

            left_blocks = []
            # List the blocks that were not merged
            left_blocks = [partition[k] for k in range(len(partition)) if k != i and k != j]

            for partition_left in left_blocks:
                if check_nested_edges(create_edges(merged_block), create_edges(partition_left)):  # Yes, skip this (i,j)
                    possible_merge.remove((i, j))
                    break
            else:
                New_nonnesting_Partition = sorted(copy.deepcopy(left_blocks + [merged_block]), key=lambda subset: subset[0])
                partition = copy.deepcopy(New_nonnesting_Partition)
                NoNonestingCreated = False
            # Add this pair to the tried set
    return partition


def Hard_Partition_NN(d, k):
    # Initialize a list of k empty classes
    partition = [[] for _ in range(k)]

    # Distribute elements into the k classes
    for i in range(k):
        partition[i] = [i + c * k for c in range(d // k + 1) if i + c * k < d]

    return partition


def Random_Sparse_Vector(d, k):
    if k > d:
        raise ValueError("k cannot be greater than d")

    # Step 1: Initialize vector
    vector = np.zeros(d)

    # Step 2: Select k unique indices
    indices = np.random.choice(d, k, replace=False)

    # Step 3: Assign random values between 0 and 10 to these indices
    vector[indices] = np.random.uniform(0, 1, k)

    return vector

#partition = [[1],[2],[3],[4],[5]]
#partition = [[1,2,3],[4,5]]
#partition = [[1,2,6],[3,4,5],[7]]
#partition = [[0],[1],[2],[3],[4],[5],[6]]
#n = 7

#[[0, 5, 11], [1, 9], [2, 10, 13, 14], [3], [4], [6], [7], [8, 12]] This partition cannot be further coarsen?
#print(Coarsen_Partition_NN([[0, 5, 11], [1, 9], [2, 10, 13, 14], [3], [4], [6], [7], [8, 12]], 15))

#print(Random_Partition_NN(40,4))
#print(is_nested([1,2],[7,9]))
#print(hard_Partition_NN(15,2))