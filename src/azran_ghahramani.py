from math import ceil
from math import floor
import numpy as np
import pickle
import random

# TODO: I don't want any partitions returned with n_clusters = 1 ... that's trash
# TODO: Should I incorporate the max_eigengaps plot thing into the main functionality?
# Seems like something I always want right? And it's faster to compute once
# TODO: Remove the if False: statements. Think we're over that now
# TODO: Remove the Q output from multiscale-k-prototypes. We don't care about that


def mean(inp):
    return sum(inp) / len(inp)


def make_distances(points, metric):
    n_points = len(points)    
    distances = [
        [metric(points[i], points[j]) for j in range(i+1, n_points)]
        for i in range(n_points-1)
    ]
    return distances


def make_W_D(points, metric, similarity):
    '''
    Args:
    points - iterable of points
    metric - distance function between two points
    similarity - non-negative monotonically decreasing function giving similarity from distance

    Returns:
    W, D
    '''

    distances = make_distances(points, metric)
    return make_W_D_from_distances(
        n_points = len(points),
        similarity = similarity,
        distances = distances
    )


def make_W_D_from_distances(n_points, similarity, distances, self_similarity=1):
    # Make W matrix
    W = np.empty((n_points, n_points))

    for i in range(n_points):
        for j in range(i, n_points):
            if i == j:
                W[i][j] = self_similarity
            else:
                distance = distances[i][j-i-1]
                W[i][j] = similarity(distance)
                W[j][i] = W[i][j]
    
    # Make D matrix from row sums
    D = np.diag([W[i].sum() for i in range(n_points)])

    # W is assumed to be full rank
    assert np.linalg.matrix_rank(W) == len(W), 'W is not full rank'
    return W, D


def kl_divergence(a, b):
    '''Computes KL-divergence'''
    if len(a) != len(b):
        raise ValueError(f'Expected inputs to kl_divergence to be of the same length, ' 
            'got {len(a)} and {len(b)}')

    return (a * np.log(a/b)).sum()


def equal_contents(ls1, ls2):
    if len(ls1) != len(ls2):
        return False
    for el in ls1:
        if el not in ls2:
            return False

    return True

def K_prototypes(transition_matrix, n_clusters, prototypes_init, stop_condition=0):
    '''
    Returns:
    partition, Q
    '''

    n_rows, n_cols = transition_matrix.shape
    prototypes = prototypes_init

    counter = 0
    previous_partition = None
    while True:
        partition = [[] for _ in range(n_clusters)] 
        
        for row in range(n_rows):
            divergences = [
                kl_divergence(
                    transition_matrix[row], 
                    prototypes[k]
                ) 
                for k in range(n_clusters)
            ]
            closest_cluster = np.argmin(divergences)
            partition[closest_cluster].append(row)

        new_prototypes = np.empty((n_clusters, n_cols))
        for k in range(n_clusters):
            # NB: Trying out a new method here of keeping old row if no partition elements
            if len(partition[k]) == 0:
                new_prototypes[k] = prototypes[k]
            else:
                new_prototypes[k] = mean([transition_matrix[m] for m in partition[k]])
        
        if previous_partition is None:
            previous_partition = partition.copy()
        else:
            if equal_contents(previous_partition, partition):
                break
            previous_partition = partition.copy()

        if counter > 5000:
            print('*** BREAKING DUE TO EXCESSIVE COUNT ***')
            break

        prototypes = new_prototypes
        counter += 1

    return partition, prototypes


def star_shaped_init(transition_matrix, n_clusters):
    '''Computes star-shaped initialization of Q (Algorithm 2)

    Outputs: Initial Q to use in K-prototypes algorithm
    
    The first row of the output is the mean of the transition matrix
    rows, then recursively the next row is taken to be the row of the
    transition matrix which maximises the minimum KL-divergence between
    previously-entered rows.
    '''

    n_points = len(transition_matrix)
    Q = np.empty((n_clusters, n_points))
    Q[0] = mean([transition_matrix[n] for n in range(n_points)])

    for k in range(1, n_clusters):
        min_divergences = []
        for n in range(n_points):
            divergences = [kl_divergence(transition_matrix[n], Q[j]) for j in range(k)]
            min_divergences.append(min(divergences))
        z = np.argmax(min_divergences)
        Q[k] = transition_matrix[z]

    return Q


def random_init(transition_matrix, n_clusters):
    n_points = len(transition_matrix)
    Q = np.empty((n_clusters, n_points))
    chosen_rows = random.sample(range(len(transition_matrix)), n_clusters)

    for k in range(n_clusters):
        Q[k] = transition_matrix[chosen_rows[k]]

    return Q



def closest_even_integer(x):
    '''
    Computes closest even integer to an input float or integer.
    If x is an integer, the output is x+1 if x is odd else x
    '''
    if not isinstance(x, (int, float)):
        raise ValueError(f'Expected {x} to be an integer or a float')

    if isinstance(x, int):
        return x if x%2 == 0 else x+1

    lower = floor(x)
    upper = ceil(x)

    # If the closest integer to x is smaller than x
    if abs(lower - x) < abs(upper - x):
        return lower if lower%2 == 0 else upper
    return upper if upper%2 == 0 else lower


def delta_k_t(k, t, eigenvalues):
    '''Equation (11). Compute t-th order eigengap between eigenvalue k and k+1'''
    if k < 0:
        raise ValueError('Input k to delta_k_t() must be a positive integer')
    if k >= len(eigenvalues)-1:
        print('k: ', k)
        print('len(eigenvalues)-1: ', len(eigenvalues)-1)

        raise ValueError('Received an input k value to delta_k_t() larger than number of eigenvalues')
    
    this_evalue = eigenvalues[k]
    next_evalue = eigenvalues[k+1]

    return abs(this_evalue)**t - abs(next_evalue)**t
    # return pow(abs(this_evalue), t) - pow(abs(next_evalue), t)


def maximal_eigengap(t, eigenvalues):
    '''Equation (14). Compute the maximal t-th order eigengap'''
    eigengaps = [delta_k_t(k, t, eigenvalues) for k in range(len(eigenvalues)-1)]
    return max(eigengaps)


def n_clusters_best_revealed(t, eigenvalues, max_clusters=None):
    '''Equation (15). Find the number of clusters best revealed by t steps'''
    if max_clusters is None:
        max_clusters = len(eigenvalues)-1

    delta_k_values = [delta_k_t(k, t, eigenvalues) for k in range(max_clusters)]

    return np.argmax(delta_k_values) + 1


def write_maximal_eigengaps_from_W_D(
    W,
    D,
    min_t,
    max_t,
    folder_path,
    filename,
    logscale=True,
    n_points=1000
    ):
    
    transition_matrix = np.linalg.inv(D).dot(W)

    evalues, _ = np.linalg.eig(transition_matrix)
    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]

    if logscale:
        t_values = np.logspace(start=min_t, stop=max_t, num=n_points)
    else:
        t_values = range(min_t, max_t+1)

    output = []

    for t in t_values:
        val = maximal_eigengap(t, evalues)
        entry = f'{t} {val}'
        if t != max_t:
            entry += '\n'
        output.append(entry)

    with open(folder_path+filename, 'w+') as fp:
        fp.writelines(output)
    

def write_maximal_eigengaps(points, metric, similarity, 
                            min_t, max_t, folder_path, filename, 
                            logscale=True, n_points=1000):

    W, D = make_W_D(points, metric, similarity)

    write_maximal_eigengaps_from_W_D(
        W, D, min_t, max_t, folder_path, 
        filename, logscale, n_points
    )


def get_eigengap_values(t_values, evalues, max_clusters=None, ignore_one_clustering=False):
    eigengap_values = {}
    max_n_clusters = len(evalues) - 1
        
    if max_clusters is not None:
        max_n_clusters = min(max_n_clusters, max_clusters)

    clusters = range(1, max_n_clusters)
    if ignore_one_clustering:
        clusters = range(2, max_n_clusters)
    
    for k in clusters:
        eigengap_values[k] = {t: delta_k_t(k-1, t, evalues) for t in t_values}
    
    return eigengap_values


def get_maximal_eigengap_information(eigengap_values):
    ex_key = list(eigengap_values.keys())[0]
    clusters = eigengap_values.keys()
    t_values = eigengap_values[ex_key].keys()
    
    max_eigengap_values = {}
    maximum_attained = {}
    
    for t in t_values:
        vals = {k: eigengap_values[k][t] for k in clusters}.items()
        cluster, maximum = max(vals, key=lambda x: x[1])
        if cluster not in maximum_attained:
            maximum_attained[cluster] = {
                'suitability': maximum,
                'n_steps': int(t),
            }
        else:
            if maximum > maximum_attained[cluster]['suitability']:
                maximum_attained[cluster] = {
                    'suitability': maximum,
                    'n_steps': int(t),
                }
                
        max_eigengap_values[t] = maximum
        
    return max_eigengap_values, maximum_attained


def multiscale_k_prototypes_from_W_D(points, W, D, max_clusters=None):
    transition_matrix = np.linalg.inv(D).dot(W)

    evalues, evectors = np.linalg.eig(transition_matrix)
    evectors = evectors.T
    evectors = np.array([evectors[n] for n in range(len(evectors))])

    # Sort eigenvalues from largest to smallest; update eigenvectors
    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]
    evectors = evectors[idx]

    if False:
        # Generate idempotent-orthogonal basis of P^t
        basis = []
        for evec in evectors:
            constant = 1 / (evec.dot(D).dot(evec))
            matrix = np.outer(evec, evec).dot(D)
            matrix *= constant
            basis.append(matrix)

    t_values = np.logspace(start=0, stop=14, num=1000)
    eigengap_values = get_eigengap_values(t_values, evalues, max_clusters)
    max_egap_values, max_attained = get_maximal_eigengap_information(eigengap_values)

    results = []

    for n_clusters in max_attained:
        if n_clusters == 1:
            continue

        n_steps = max_attained[n_clusters]['n_steps']
        suitability = max_attained[n_clusters]['suitability']
        
        transition_matrix_power = np.linalg.matrix_power(transition_matrix, n_steps)

        Q_init = star_shaped_init(transition_matrix_power, n_clusters)
        partition, Q = K_prototypes(transition_matrix_power, n_clusters, Q_init)
        results_entry = dict(
            n_clusters = n_clusters,
            n_steps = n_steps,
            suitability = suitability,
            partition = partition,
        )
        results.append(results_entry)

    return results


def multiscale_k_prototypes(points, metric, similarity, max_clusters=None):
    '''
    Args:
    As in make_W_D()
    '''

    W, D = make_W_D(points, metric, similarity)
    return multiscale_k_prototypes_from_W_D(points, W, D, max_clusters)
