from itertools import product, combinations
from math import factorial
import cvxpy as cp
import numpy as np

I = np.matrix([[1,0],[0,1]])
X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])
Z = np.matrix([[1,0],[0,-1]])

pauli_strings = ['I','X','Y','Z']
nt_pauli_strings = ['X','Y','Z']
pauli_matrices = [I,X,Y,Z]

def ind_to_pauli_name(i):
    if i == 0:
        return 'I'
    elif i == 1:
        return 'X'
    elif i == 2:
        return 'Y'
    else:
        return 'Z'

def comb(n,k):
    return factorial(n) / (factorial(k)* factorial(n-k))


def tensor_prod(arr):
    out = arr[0]
    for mat in arr[1:]:
        out = np.kron(out, mat)
    return out

def symmetrize_mat(mat):
    # copies upper triangular into lower triangular
    for i in range(n):
        for j in range(i+1,n):
            mat[j][i] = mat[i][j]
    return mat

def one_body_op(op, i):
    ops = [(op if k ==i else I) for k in range(n)]
    return tensor_prod(ops)

def two_body_op(op, i, j):
    ops = [(op if k ==i or k==j else I) for k in range(n)]
    return tensor_prod(ops)

def qmc_term(w, i, j):
    return w * (two_body_op(I, i, j) - two_body_op(X, i, j) - two_body_op(Y, i, j) - two_body_op(Z, i, j))



n = 4 # number of qubits
adj_mat = [[0 for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(i+1,n):
        adj_mat[i][j]=np.random.rand()
        #if j - i == 1:
        #    adj_mat[i][j] = 1
        #entry = np.random.rand()
        #adj_mat[i][j] = entry


adj_mat = symmetrize_mat(adj_mat)
adj_mat = np.array(adj_mat)

print(adj_mat)


def get_pstring(subset, paulis):
    assert len(subset) == len(paulis)
    out = []
    for i in range(n):
        try:
            out.append(paulis[subset.index(i)])
        except ValueError:
            out.append('I')
    return out

def get_p_prod(p1, p2):
    # return (phase, Pauli)
    if p1 == 'I':
        return (1, p2)
    if p2 == 'I':
        return (1, p1)
    if p1 == p2:
        return (1,'I')
    if (p1 == 'X' and p2 == 'Y'):
        return (1j,'Z')
    if (p1 == 'X' and p2 == 'Z'):
        return (-1j,'Y')
    if (p1 == 'Y' and p2 == 'X'):
        return (-1j,'Z')
    if (p1 == 'Y' and p2 == 'Z'):
        return (1j,'X')
    if (p1 == 'Z' and p2 == 'X'):
        return (1j,'Y')
    if (p1 == 'Z' and p2 == 'Y'):
        return (-1j,'X')
    assert 1 == 0


def get_pstring_prod(ps1, ps2):
    assert len(ps1) == len(ps2) == n
    phase = 1
    prod = []
    for i in range(n):
        (ph, p) = get_p_prod(ps1[i], ps2[i])
        prod.append(p)
        phase *= ph
    return (phase, prod)


def get_lasserre_value(l, n, adj_mat):
    
    #get mat_dim 
    mat_dim = 0
    for deg in range(l+1):
        mat_dim += int(comb(n,deg)*(3**deg))

    # create maps between pauli string and index
    pauli_to_index = {}
    index_to_pauli = {}

    istring = tuple(['I']*n)
    pauli_to_index[istring] = 0
    index_to_pauli[0] = istring

    i=1
    for deg in range(1,l+1):
        for subset in combinations(range(n),deg):
            for l_body_pauli in product(nt_pauli_strings, repeat=deg):
                pstring = get_pstring(subset, l_body_pauli)
                pstring = tuple(pstring)
                pauli_to_index[pstring] = i
                index_to_pauli[i] = pstring
                i += 1
    assert len(pauli_to_index) == mat_dim

    C = np.array([[0.0 for _ in range(mat_dim)] for _ in range(mat_dim)])
    # compute C entries given adj_mat
    for subset in combinations(range(n),2):
        for l_body_pauli in product(pauli_strings, repeat=2):
            pstring = tuple(get_pstring(subset, l_body_pauli))
            if l_body_pauli == ('I','I'):
                C[pauli_to_index[pstring]][0] += adj_mat[subset[0]][subset[1]]
            elif l_body_pauli == ('X','X') or l_body_pauli == ('Y','Y') or l_body_pauli == ('Z','Z'):
                C[pauli_to_index[pstring]][0] -= adj_mat[subset[0]][subset[1]]


    #define constraints
    M = cp.Variable((mat_dim, mat_dim), symmetric=True)
    cM = cp.Variable((mat_dim, mat_dim), complex=True)
    constraints = [M >> 0]
    cconstraints = [cM >> 0, cM.H == cM]


    for i1 in range(mat_dim):
        for j1 in range(i1, mat_dim):
            if i1 == j1:
                # all diagonals are 1
                constraints.append( M[i1][j1] == 1)
                cconstraints.append( cM[i1][j1] == 1 )
                continue

            (phase1, pauli1) = get_pstring_prod(index_to_pauli[i1], index_to_pauli[j1])
            if phase1 == 1j or phase1 == -1j:
                # the paulis anticommute
                constraints.append( M[i1][j1] == 0 )
                cconstraints.append( cp.real(cM[i1][j1]) == 0 )
                #continue
            else:
                # the Paulis commute
                cconstraints.append( cp.imag(cM[i1][j1]) == 0 )
                
            for i2 in range(mat_dim):
                j2_pauli = tuple( get_pstring_prod(pauli1, index_to_pauli[i2])[1] )
                if not j2_pauli in pauli_to_index:
                    continue
                j2 = pauli_to_index[j2_pauli] # hehe sorry for this stupid code 
                (phase2, pauli2) = get_pstring_prod(index_to_pauli[i2], j2_pauli)

                assert (pauli1 == pauli2) 
                if (phase1 == 1 or phase1 == -1) and (phase2 == 1 or phase2 == -1):
                    # both pairs of Paulis commute with each other
                    constraints.append( M[i1][j1] == (np.real(phase1) * np.real(phase2)) * M[i2][j2] )
                cconstraints.append( phase2 * cM[i1][j1] == phase1 * cM[i2][j2] )

                

                '''
                for j2 in range(mat_dim):
                    (phase2, pauli2) = get_pstring_prod(index_to_pauli[i2], index_to_pauli[j2])
                    if (phase2 == 1 or phase2 == -1) and pauli1 == pauli2:
                        assert (phase1 == 1 or phase1 == -1)
                        constraints.append( M[i1][j1] == (np.real(phase1) * np.real(phase2)) * M[i2][j2] )
                '''

    print('constraints loaded')

    prob = cp.Problem(cp.Maximize(cp.trace(C @ M)), constraints)
    #prob.solve()
    prob.solve(solver = cp.MOSEK)

    print('solved real Lasserre')

    cprob = cp.Problem(cp.Maximize(cp.real(cp.trace(C @ cM))), cconstraints)
    #cprob.solve()
    cprob.solve(solver = cp.MOSEK)

    print('solved complex Lasserre')

    return (prob.value, cprob.value)

    #return (prob.value, cprob.value)

def get_diagonalized_value(adj_mat):
    H = None
    for j in range(n):
        for i in range(j):
            if H is None:
                H = qmc_term(adj_mat[i][j], i, j)
            else:
                H += qmc_term(adj_mat[i][j], i, j)

    return sorted(np.linalg.eigvals(H))[-1]


print(get_lasserre_value(2, n, adj_mat))
print(get_diagonalized_value(adj_mat))
