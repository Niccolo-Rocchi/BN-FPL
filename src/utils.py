from fractions import Fraction
from tempfile import TemporaryDirectory

import cdd
import cdd.gmp
import cvxpy as cp
import hopsy
import numpy as np
import pyagrum as gum
import pandas as pd

from src.config import safe_assert


# Create the BN storing the counts of events
def get_bn_counts(bn, data):

    # Init the BN
    bn_counts = gum.BayesNet(bn)

    for node in bn.names():

        cpt = bn.cpt(node)

        if len(bn.parents(node)) != 0:
            var_size = bn.variable(node).domainSize()
            n_rows = cpt.domainSize() // var_size

            index = cpt.topandas().index
            index_df = index.to_frame(index=False)

            subset=index_df.columns.tolist()
            data_counts = data.value_counts(subset=subset).reset_index(name='counts')
            res = pd.merge(index_df, data_counts, on=subset, how='left')
            res['counts'] = res['counts'].fillna(0).astype(int)

            cpt_resh = cpt[:].reshape(n_rows, var_size)
            cpt_new = np.round(cpt_resh * np.atleast_2d(res["counts"]).T)

        else:
            cpt_new = np.round(cpt[:] * len(data))

        bn_counts.cpt(node).fillWith(cpt_new.flatten().tolist())

        # Debug
        safe_assert(np.sum(cpt_new) == len(data))

    return bn_counts

# Get a bidimensional CPT
def get_tabular_cpt(cpt) -> np.array:

    cpt = np.atleast_2d(cpt[:])
    if cpt.ndim == 2: return cpt

    var_size, n_rows = get_cpt_shape(cpt)

    return cpt.reshape(n_rows, var_size)

# Get the shape of a BN's CPT
def get_cpt_shape(cpt) -> tuple:

    cpt = np.atleast_2d(cpt[:])
    var_size = cpt.shape[0]
    n_rows = np.prod(cpt.shape[1:]).astype(int)

    return var_size, n_rows

# Find the index in a CPT corresponding to a specific configuration of the parents
def get_cpt_index(bn: gum.BayesNet, var:str, parents:dict = None):

    ''' 
    Notice: the CPT is thought as a bidimensional matrix.
    '''
    if parents is None: return 0
    index = bn.cpt(var).topandas().index
    index_df = index.to_frame(index=False)
    query_str = " and ".join([f"{col} == @parents['{col}']" for col in parents])
    res_query = index_df.query(query_str)
    if res_query.empty: 
        raise RuntimeError(f"Wrong parent configuration for variable '{var}'.")
    
    return res_query.index[0]

# Get the BN inside a CN with max entropy distribution
def maxent_cn(bn_min, bn_max) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # For each variable ...
    for var in bn.names():

        # ... get the maxent CPT, ...
        cpt = maxent_cpt(bn_min.cpt(var), bn_max.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the BN CPT inside a CN CPT with max entropy distribution
def maxent_cpt(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the maxent credal set, ...
        c = maxent_cset(cpt_min[row, :], cpt_max[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the max-entropy distribution inside a credal set
def maxent_cset(vec_min, vec_max) -> np.array:

    rank = {v: k for k, v in enumerate(sorted(set(vec_min)))}
    vec_order = np.array([rank[val] for val in vec_min])
    s = 1 - np.sum(vec_min)
    out = vec_min

    while s > 0:
        idx0 = np.where(vec_order == 0)[0]
        idx1 = np.where(vec_order == 1)[0]
        idx0_len = len(idx0)
        idx1_len = len(idx1)

        if idx1_len != 0:
            diff = out[idx1[0]] - out[idx0[0]]
            s_cond = s / idx0_len < diff
            mat = np.stack(
                [
                    (
                        (s / idx0_len) * np.ones(len(idx0))
                        if s_cond
                        else diff * np.ones(len(idx0))
                    ),
                    vec_max[idx0] - out[idx0],
                ]
            )
        else:
            s_cond = True
            mat = np.stack(
                [(s / idx0_len) * np.ones(len(idx0)), vec_max[idx0] - out[idx0]]
            )

        mat_min = np.min(mat)
        q = np.argwhere(mat == mat_min)

        if np.any(q[:, 0] == 1):
            if len(idx0) > len(q):
                vec_order[~np.isin(np.arange(len(out)), idx0[q[:, 1]])] += 1
        elif not s_cond:
            vec_order[idx0[q[:, 1]]] += 1

        out[idx0] += mat_min
        s -= mat_min * len(idx0)
        vec_order -= 1

    return out


# Get the max likelihood BN inside a CN
def mle_cn(bn_min, bn_max, data) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # Store counts
    bn_counts = get_bn_counts(bn, data)

    # For each variable ...
    for var in bn.names():

        # ... get the MLE CPT, ...
        cpt = mle_cpt(bn_min.cpt(var), bn_max.cpt(var), bn_counts.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the max likelihood BN CPT inside a CN CPT
def mle_cpt(cpt_min, cpt_max, cpt_counts) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())
    cpt_counts = np.atleast_2d(cpt_counts.topandas())

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the MLE credal set, ...
        c = mle_cset(cpt_min[row, :], cpt_max[row, :], cpt_counts[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the max likelihood distribution inside a credal set
def mle_cset(vec_min, vec_max, counts) -> np.array:

    # Number of variables to optimize
    n_par = len(vec_min)
    p = cp.Variable(n_par)

    # Log-likelihood to maximize
    objective = cp.Maximize(counts @ cp.log(p))

    # Constraints
    constraints = [cp.sum(p) == 1, p >= np.maximum(vec_min, 10e-9), p <= vec_max]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)

    mle_vec = np.array(p.value)

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return mle_vec


# Get the max (-likelihood) BN (MNE) within a CN, i.e., the min likelihood BN.
def mne_cn(bn_min, bn_max, data) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # Store counts
    bn_counts = get_bn_counts(bn, data)

    # For each variable ...
    for var in bn.names():

        # ... get the MNE CPT, ...
        cpt = mne_cpt(bn_min.cpt(var), bn_max.cpt(var), bn_counts.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the MNE BN CPT inside a CN CPT
def mne_cpt(cpt_min, cpt_max, cpt_counts) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())
    cpt_counts = np.atleast_2d(cpt_counts.topandas())

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the MNE credal set, ...
        c = mne_cset(cpt_min[row, :], cpt_max[row, :], cpt_counts[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the MNE distribution inside a credal set
def mne_cset(vec_min, vec_max, counts) -> np.array:

    # Get the credal set vertices
    vertices = vertices_cset(vec_min, vec_max)

    # Get the vertex that has the maximum (-likelihood)
    vec_best = vertices[0, :]
    mne_best = counts @ -np.log(np.where(vec_best > 0, vec_best, 1))

    for row in range(vertices.shape[0]):

        vec = vertices[row, :]
        mne = counts @ -np.log(np.where(vec > 0, vec, 1))

        if mne > mne_best:
            mne_best = mne
            vec_best = vec

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return vec_best


# Get a random BN inside a CN
def ran_cn(bn_min, bn_max) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # For each variable ...
    for var in bn.names():

        # ... get a random CPT, ...
        cpt = ran_cpt(bn_min.cpt(var), bn_max.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get a random BN CPT inside a CN CPT
def ran_cpt(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... sample randomly from the credal set, ...
        c = ran_cset(cpt_min[row, :], cpt_max[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get a random distribution inside a credal set
def ran_cset(vec_min, vec_max) -> np.array:

    # Get the credal set vertices
    vertices = vertices_cset(vec_min, vec_max)

    # Sample weights for vertices
    n = len(vertices)
    w = np.random.dirichlet(np.ones(n))

    # Draw the linear combination of vertices
    ran_vec = w @ vertices

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return ran_vec


# Get the centroid of a CN
def centroid_cn(bn_min, bn_max) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # For each variable ...
    for var in bn.names():

        # ... get the centroid CPT, ...
        cpt = centroid_cpt(bn_min.cpt(var), bn_max.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the centroid of a CN CPT
def centroid_cpt(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the centroid credal set, ...
        c = centroid_cset(cpt_min[row, :], cpt_max[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the centroid of a credal set as the average of its extreme points
def centroid_cset(vec_min, vec_max) -> np.array:

    # Get the credal set vertices
    vertices = vertices_cset(vec_min, vec_max)

    # Compute the centroid as the average across extreme points
    centroid = np.sum(vertices, axis=0) / len(vertices)

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return centroid


# Get the credal set vertices
def vertices_cset(vec_min, vec_max) -> np.array:

    # Define the (in)equalities (i.e., get the H-representation of the credal set)
    n_par = len(vec_min)
    A = np.concatenate(
        (-np.eye(n_par), np.eye(n_par), np.atleast_2d(np.ones(n_par))), axis=0
    )
    b = np.concatenate((vec_max, -vec_min, np.atleast_1d(-1))).reshape(len(A), 1)
    bA = np.concatenate((b, A), axis=1)
    bA_frac = np.array(
        [[Fraction(x).limit_denominator() for x in row] for row in bA], dtype=object
    )  # Needed for numerical stability
    mat_frac = cdd.gmp.matrix_from_array(
        array=bA_frac, rep_type=cdd.RepType.INEQUALITY, lin_set=set([len(A) - 1])
    )

    # Get the polytope and extreme points. Each point is a row of the matrix `vertices`
    poly_frac = cdd.gmp.polyhedron_from_matrix(mat_frac)
    ext_frac = cdd.gmp.copy_generators(poly_frac)
    vertices_frac = np.array(ext_frac.array)[:, 1:]
    vertices = np.array([[float(x) for x in row] for row in vertices_frac], dtype=float)

    # Debug
    safe_assert(len(b) == 2 * len(vec_min) + 1)
    safe_assert(A.shape == (len(b), len(vec_min)))
    safe_assert(bA.shape == (2 * len(vec_min) + 1, len(vec_min) + 1))
    safe_assert(vertices.shape[1] == n_par)

    return vertices


# BNs sampler from a CN
def sample_from_cn(bn_min, bn_max, n_bns: int) -> list:

    # Get the DAG and extreme BNs
    dag = gum.BayesNet(bn_min)

    # For each variable ...
    cpts_dict = {}
    for var in dag.names():

        # ... sample `n_bns` CPTs from the CN
        cpts_dict[var] = sample_from_cpts(bn_min.cpt(var), bn_max.cpt(var), n_bns)

    # For each sample ...
    bns = []
    for i in range(n_bns):

        # ... init an empty BN ...
        bn = gum.BayesNet(dag)

        # ... and fill its CPTs
        for var in dag.names():
            bn.cpt(var).fillWith(cpts_dict[var][i])

        bns.append(bn)

        # Debug
        safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    # Debug
    safe_assert(len(cpts_dict) == len(dag.names()))
    safe_assert(len(bns) == n_bns)

    return bns


# Sample from two extreme CPTs
def sample_from_cpts(cpt_min, cpt_max, n_bns) -> list:

    # Transform CPTs into pandas dataframes
    cpt_min = np.atleast_2d(cpt_min.topandas())
    cpt_max = np.atleast_2d(cpt_max.topandas())

    # For each row in the CPT ...
    credal_dict = {}
    for row in range(cpt_min.shape[0]):

        # ... sample `n_bns` points from the credal set
        credal_dict[row] = sample_from_cset(cpt_min[row, :], cpt_max[row, :], n_bns)

    # For each sample ...
    cpt_samples = []
    for i in range(n_bns):

        # ... build the CPT
        cpt = []
        for row in range(cpt_min.shape[0]):
            cpt.append(credal_dict[row][i])

        cpt = np.array(cpt).flatten()
        cpt_samples.append(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(len(credal_dict) == cpt_min.shape[0])
    safe_assert(len(cpt_samples) == n_bns)

    return cpt_samples


# Sample from a credal set K(x | pi_x), i.e., a constrained polytope.
def sample_from_cset(vec_min, vec_max, n_bns) -> list:
    """
    We assume a credal set is a polytope in a space of #X parameters, defined by a:
     - Multi-dimensional rectangle, i.e., inequality constraint Ax <= b, and
     - Hyperplane (provided all the variables sum up to 1), i.e., equality constraint A_eq x = b_eq.
    This is true if the CN has been learnt by local IDM, for instance.
    """

    # Define the rectangle
    n_par = len(vec_min)
    A = np.concatenate((np.eye(n_par), -np.eye(n_par)), axis=0)
    b = np.concatenate((vec_max, -vec_min))
    rectangle = hopsy.Problem(A=A, b=b)

    # Define the hyperplane
    A_eq = np.array([np.ones(n_par)])
    b_eq = np.array([1.0])

    # Define the polytope as a constrained rectangle (i.e., get the H-representation of the credal set)
    constrained_rectangle = hopsy.add_equality_constraints(
        rectangle, A_eq=A_eq, b_eq=b_eq
    )

    # Sample from the polytope
    mc = hopsy.MarkovChain(constrained_rectangle)
    rng = hopsy.RandomNumberGenerator(42)
    _, constrained_samples = hopsy.sample(mc, rng, n_bns, thinning=10)
    constrained_samples = constrained_samples[0]

    # Debug
    safe_assert(np.all(vec_min <= vec_max))
    safe_assert(n_par == len(vec_max))
    safe_assert(n_par == A.shape[1])
    safe_assert(n_par == A_eq.shape[1])
    safe_assert(len(constrained_samples) == n_bns)
    for i in constrained_samples:
        safe_assert(len(i) == n_par)

    return constrained_samples


# Check the consistency of a BN as sampled from a CN. Returns the number of inconsistent CPTs.
def check_consistency(bn, bn_min, bn_max, verbose=False) -> int:

    n_issues = 0

    for var in bn.names():
        bn_cpt = np.atleast_2d(bn.cpt(var).topandas())
        bn_min_cpt = np.atleast_2d(bn_min.cpt(var).topandas())
        bn_max_cpt = np.atleast_2d(bn_max.cpt(var).topandas())

        # Check if probabilities sum to 1
        sum_vec = np.sum(bn_cpt, axis=1)
        probability_consistency = np.all(np.abs(sum_vec - 1) < 1e-5)

        # Check if the BN CPT is >= min CPT
        min_consistency = np.all(bn_cpt - bn_min_cpt >= -1e-5)

        # Check if the BN CPT is <= max CPT
        max_consistency = np.all(bn_cpt - bn_max_cpt <= 1e-5)

        consistency = probability_consistency and min_consistency and max_consistency

        if consistency:
            continue
        else:
            n_issues += 1
            if verbose:
                print("Variable: ", var)
                print("probability_consistency: ", probability_consistency)
                print("min_consistency: ", min_consistency)
                print("max_consistency: ", max_consistency)
                print("BN CPT: ")
                print(bn_cpt)
                print("BN min CPT: ")
                print(bn_min_cpt)
                print("BN max CPT: ")
                print(bn_max_cpt)

    return n_issues


# Extract BN min and BN max from a CN
def get_min_max_bns(cn, exp: str = ""):

    with TemporaryDirectory() as tmp_path:
        cn.saveBNsMinMax(f"{tmp_path}/bn_min_{exp}.bif", f"{tmp_path}/bn_max_{exp}.bif")
        bn_min = gum.loadBN(f"{tmp_path}/bn_min_{exp}.bif")
        bn_max = gum.loadBN(f"{tmp_path}/bn_max_{exp}.bif")

    return bn_min, bn_max


# Generate a random CN with local IDM
def generate_random_cn(n_nodes, edge_density, n_modmax, ess, s_size) -> tuple:

    # Generate a BN
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(
        n_nodes=n_nodes, n_arcs=int(n_nodes * edge_density), n_modmax=n_modmax
    )

    # Generate data
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(s_size)
    data = data_gen.to_pandas()

    # Learn the CN by local IDM
    bn_counts = get_bn_counts(bn, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(ess)

    return cn, data


# Get a value from a BN's CPT
def cpt_value(
    bn: gum.BayesNet, x_var: str, x_value: float, parents: dict = None
) -> float:
    """
    Get P(X=x | parents) from the BN's CPT of X.
    `x_var` is the X name, while `x_value` is x.
    """

    cpt = bn.cpt(x_var)
    inst = gum.Instantiation(cpt)
    inst[x_var] = x_value

    if parents:
        for var in parents.keys():
            inst[var] = parents[var]
        safe_assert(bn.parents(x_var) == set(bn.ids(parents.keys())))
    else:
        safe_assert(len(bn.parents(x_var)) == 0)

    return max(cpt.get(inst), 1e-10)  # Smoothing


# Learn BN parameters from a given BN and data
def learn_bn_params(bn, data):

    bn_copy = gum.BayesNet(bn)    

    learner = gum.BNLearner(data)
    learner.useSmoothingPrior(1e-10)
    bn_learnt = learner.learnParameters(bn_copy)

    return bn_learnt

# Extract a subgraph from a given BN
def get_subgraph(bn, vars_to_keep: set):

    sub = gum.BayesNet(bn)
    for var in set(bn.names()) - vars_to_keep:
        sub.erase(var)
    
    return sub
