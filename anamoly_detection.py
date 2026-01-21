#import cirq
import numpy as np
import scipy.linalg
from scipy.stats import unitary_group
import matplotlib.pyplot as plt

class SETUP:
    #runN times for population of size N for random sampling, later add constarint that via conjugation with U of
    #|0><0| leaves it unchanged, later if using in other basis change conjugation with U of |a><a|
    @staticmethod
    def chromo_gen(N):
        U = unitary_group.rvs(N)
        detU = np.linalg.det(U)
        U = U * np.exp(-1j * np.angle(detU) / N)
        return U

    #since i need to comit to an origin using U_ref, choose accordingly
    @staticmethod
    def linearisation(U, U_ref):
        return scipy.linalg.logm(U_ref.conj().T @ U)

    #F(nxm) is isomorphic to R(2nm), did for stat analysis convevnience
    @staticmethod
    def vectorisation(A):
        v = A.flatten()
        return np.concatenate([v.real, v.imag])

    #used for physical interpretaion and track evolution
    #do everytime crossover is performed ofc
    @staticmethod
    def gene_extract(obs_list, U):
        g = []
        n = U.shape[0]
        ket0 = np.zeros((n, 1), dtype=complex)
        ket0[0, 0] = 1.0
        for O in obs_list:
            val = (ket0.conj().T @ U.conj().T @ O @ U @ ket0).item()
            g.append(np.real(val))
        return np.array(g)
    #later replace with cirq

def fitness_func(g):
    return float(np.linalg.norm(g))

#P has linerased and flattened si that stats analysis can be performed
def mean_covariance(P):
    P = np.array(P)
    mean = np.mean(P, axis=0)

    diff = (P[0] - mean).reshape(-1, 1)
    cov = diff @ diff.T
    for i in range(1, len(P)):
        diff = (P[i] - mean).reshape(-1, 1)
        cov += diff @ diff.T

    cov /= (len(P) - 1)
    cov += 1e-6 * np.eye(cov.shape[0])
    return mean, cov


def mahalonobis_distance(cov_inv, a, b):
    d = a - b
    return np.sqrt(d.T @ cov_inv @ d)


class Anomaly_Detector:
    def __init__(self, P):
        self.P = np.array(P)
        self.mean, self.cov = mean_covariance(self.P)
        self.cov_inv = np.linalg.pinv(self.cov)

    #dense and fit_check must both be true then form kNN cluster and compare_clusters confirm anamoly
    def denseORnot(self, threshold, theta, delta, c_max, c_min):
        d = mahalonobis_distance(self.cov_inv, theta, self.mean)
        if d <= threshold:
            return False

        c = 0
        for x in self.P:
            if not np.allclose(x, theta):
                if mahalonobis_distance(self.cov_inv, theta, x) < delta:
                    c += 1

        return (c_min < c < c_max)

    def fit_check(self, f_min, f_max, genes):
        f = fitness_func(genes)
        return (f_min < f < f_max)

    #objects for comparison : the blob which might imply an anamoly
    def kNN_cluster(self, theta, delta):
        cluster = [theta]
        for x in self.P:
            if mahalonobis_distance(self.cov_inv, theta, x) < delta:
                cluster.append(x)
        mean_c, cov_c = mean_covariance(cluster)
        return (mean_c, cov_c, len(cluster))

    #for temporal consistency if the anamoly cluster across evolution
    def compare_clusters(A,B,tau):
        mean_A, cov_A, size_A = A[0],A[1],A[2]
        mean_B, cov_B, size_B = B[0],B[1],B[2]

        covB_inv = np.linalg.pinv(cov_B)

        # Mahalanobis distance between cluster means
        d = mahalonobis_distance(covB_inv, mean_A, mean_B)

        if d > tau:
            #clusters too far, not physically interpretable perhaps idk, prolly change this acordingly
            #(same anamoly might be far from its predecessor in evolution, so use some other metric as said accordingly above)
            return False

        return True

#Observables are chosen Hermitian and traceless so they pair naturally
#with the traceless anti-Hermitian Lie algebra of SU(N), ensuring consistency with 
#the group–algebra exponential map after removing global phase.
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

U_ref = SETUP.chromo_gen(4)
#random sampling
def random_sampling(U_ref, obs_list, pop=30,N=4):
    P = []
    U_list = []
    L_list = []
    genes_list = []
    for _ in range(pop):
        U = SETUP.chromo_gen(N)
        U_list.append(U)
        L = SETUP.linearisation(U, U_ref)
        L_list.append(L)
        v = SETUP.vectorisation(L)
        P.append(v)
        

        genes = SETUP.gene_extract(obs_list, U)
        genes_list.append(genes)

    return P, genes_list,L_list,U_list

O = [np.kron(Z, Z)]
P1,g1,L1,U1_ = random_sampling(U_ref, O, pop=10, N=4)


"""Use U1_"""
def mutation(U,mut_rate = 0.01):
    #apply small random unitary perturbation
    err = SETUP.chromo_gen(U_ref.shape[0])
    err = SETUP.linearisation(err, np.eye(U_ref.shape[0]))

    #leftmatmul do that the chromsome is perturbed before application on the state(when gene vectors are derived)
    #rightmatmul corresponds to noise in the state itself, after operating each output goes to diffrent orbit
    U = scipy.linalg.expm(mut_rate*err)@U
    return U


"""use L1"""
#U1,U2 before vectorisation
def crossover(l1,l2):
    #simple average crossover in linearised space
    Lc = 0.5*(l1+l2)
    Uc = U_ref @ scipy.linalg.expm(Lc)
    return Uc
#g1 contains arrays which have scalars corr to different observable output, so fitness must be a vector function
def selection(g1, k,L1):
    fitness = np.array([fitness_func(ind) for ind in g1])
    idx = np.argsort(fitness)[::-1]
    P = [L1[i] for i in idx[:k]]
    sel = [g1[i] for i in idx[:k]]
    return P,sel#P now usable directly for corssover



def convergence():
    pass