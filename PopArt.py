from functools import wraps
from typing import List, Tuple
import mosek
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
import cvxpy as cp
import argparse
from my_catoni import *

from cvxpy.atoms import bmat, reshape, trace, upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.constraint import Constraint

'''

'''


parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="dimension of the action vector", default=10)
parser.add_argument("--s", type=int, help="sparsity of the hidden parameter vector", default=2)
parser.add_argument("--same", type=bool, help='Whether we should use equal strategy which is ours', default=False)
parser.add_argument("--inc",type=int, help='Incremental scale of the experiment', default=1000)
parser.add_argument("--T_min",type=int, help='Minimum length of experiment to examine', default=0)
parser.add_argument("--howlong",type=int, help='Maximum multiple of T_min for the experiment. For example, howlong=20 and inc=2000 means maximum 40000 rounds', default=10)
parser.add_argument("--repeat",type=int, help='How many times it will repeat for each experiment setting', default=10)
parser.add_argument("--sigma",type=float, help='Variance', default=0.1)
parser.add_argument("--delta",type=float, help='Error probability in bandit', default=0.05)
parser.add_argument("--action_set", help='which action will I use - \'hard\' is Case 1 in the figure, and \'uniform\' is the Case 2 in the figure', default='hard')
parser.add_argument("--num_of_action_set", type=int, help='Number of actions in the action set in Case 2. Case 1 has fixed number of actions', default=150),
parser.add_argument("--cheat", type=float, help='theta_0 = cheat * theta', default=0)
args = parser.parse_args()


# functions for the optimization of lasso. Basically cvxpy format functions.
def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y) ** 2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)



