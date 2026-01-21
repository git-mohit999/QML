import cirq
import numpy as np

"""
qubit_gene, qubit_chromsome
=>encode as : Ry,Rz to get samples
==>mutate by adding phase(to aid in crceation of mutilple enseblems)

fitness function

STORE in dic sort then do,
crossover(from formula)

choose then remake using formula calculate the del(t-t') acorss all if above threshold continue with crosover type shit otherwise tweak
hyperparametrs.

continue till del(t-t') tends to 0.

"""
rng = np.random.default_rng(42)

def reg():        
    a = (rng.integers(0, 101) / 100) * np.pi
    b = (rng.integers(0, 101) / 100) * 2 * np.pi

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
            cirq.ry(a)(q),
            cirq.rz(b)(q),
        )
    return circuit

def blochvector(circuit):
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    psi = res.final_state_vector   # length 2: [alpha, beta]

    alpha, beta = psi[0], psi[1]

    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2

    return [x,y,z]

N = 100
M = int(input("Enter number of genes :"))
P = []
for i in range(N):
    C = [blochvector(reg()) for i in range(M)]
    P.append(C)

"""
P = [C1 = [[q1],[q2]],
     C2 = [[q1],q2]]
"""

def fitness_func(P):
    fitval=[]
    for i in range(N):
        R = np.array(P[i])
        G = R@R.T
        D = np.sqrt(2-2*G)

        f = np.sum(1.0 / D[np.triu_indices(len(R), k=1)])
        fitval.append(f)
    
    P_tilda = [C for _,C in sorted(zip(fitval,P))]
    
    return (P_tilda,min(fitval))

#p<1
#0<m<1-p
def QCO(P,p,m):
    P_tilda,opti_val = fitness_func(P)
    del P_tilda[int(np.ceil(N/2)):N]

    Pp = P_tilda
    Pc = []

    for j in range(len(Pp)):
        C_new = []
        for i in range(M):
            a_idx = i
            b_idx = (i + 1) % M

            theta1 = np.arccos(Pp[j][a_idx][2])
            phi1   = np.arctan2(Pp[j][a_idx][1], Pp[j][a_idx][0])
            theta2 = np.arccos(Pp[j][b_idx][2])
            phi2   = np.arctan2(Pp[j][b_idx][1], Pp[j][b_idx][0])

            alpha1 = np.cos(theta1 / 2)
            beta1  = np.exp(1j * phi1) * np.sin(theta1 / 2)

            alpha2 = np.cos(theta2 / 2)
            beta2  = np.exp(1j * phi2) * np.sin(theta2 / 2)

            gamma_s = rng.uniform(0, 2*np.pi)
            theta_s = rng.uniform(0, 2*np.pi)

            alpha2_perp = -np.conj(beta2)
            beta2_perp  =  np.conj(alpha2)

            alpha = (
                np.sqrt(p) * alpha2
                + np.exp(1j * gamma_s) * np.sqrt(1 - p - m) * alpha1
                + np.exp(1j * theta_s) * np.sqrt(m) * alpha2_perp
            )

            beta = (
                np.sqrt(p) * beta2
                + np.exp(1j * gamma_s) * np.sqrt(1 - p - m) * beta1
                + np.exp(1j * theta_s) * np.sqrt(m) * beta2_perp
            )

            norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
            alpha /= norm
            beta  /= norm

            x = 2 * np.real(np.conj(alpha) * beta)
            y = 2 * np.imag(np.conj(alpha) * beta)
            z = np.abs(alpha)**2 - np.abs(beta)**2

            C_new.append([x, y, z])

        Pc.append(C_new)
    P_ = [*Pp, *Pc]
    return (P_,opti_val)


def convergence(p=0.9):
    A = 2#for treshold
    B = 2#m value
    d=True
    counter = 0
    T = 10**(-A)
    Pd,prev_mf = QCO(P,p,10**(-B))
    del_ = 1
    slow_counter =0
    MAX_SLOW = 345
    while(d):
        if counter%100==0:
            print(f"At {prev_mf} energy and {del_} difference")
      
        Pd1,mf = QCO(Pd,p,10**(-B))
        del_ = abs(prev_mf - mf)
        Pd = Pd1

        if del_ < T:
            slow_counter += 1
        else:
            slow_counter = 0 

        if slow_counter >= MAX_SLOW:
            if B >= 5:          # FINAL STAGE
                # stop: very slow + finest scale = done
                d = False
            else:
                # go to finer stage
                A += 1
                B += 1
                T = 10**(-A)
                slow_counter = 0 

        counter+=1
        prev_mf = mf
        
        
    return [mf,counter]

print(convergence(p=0.9))