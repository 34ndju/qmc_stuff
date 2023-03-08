import numpy as np

n=4

I = np.matrix([[1,0],[0,1]])
X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])
Z = np.matrix([[1,0],[0,-1]])

paulis=[I,X,Y,Z]

sqrt2 = np.sqrt(2)
Had = np.matrix([[1/sqrt2, 1/sqrt2],[1/sqrt2, -1/sqrt2]])

psi = np.array([0, n/np.sqrt(n**2+n), -1/np.sqrt(n**2+n), 0])
zeros = np.array([1,0,0,0])
rho = np.outer(psi,psi) + (n-1)/(n**2+n) * np.outer(zeros, zeros)

sigma = 0.5*rho + 0.5*np.kron(X,X).dot(rho).dot(np.kron(X,X))

def ind_to_pauli_name(i):
    if i == 0:
        return 'I'
    elif i == 1:
        return 'X'
    elif i == 2:
        return 'Y'
    else:
        return 'Z'

'''
for i in range(4):
    for j in range(4):
        print(ind_to_pauli_name(i) + ind_to_pauli_name(j), np.trace(sigma.dot(np.kron(paulis[i], paulis[j]))))
'''


def tensor_prod(arr):
    out = arr[0]
    for mat in arr[1:]:
        out = np.kron(out, mat)
    return out

def symmetrize_mat(mat):
    # copies upper triangular into lower triangular
    for j in range(n):
        for i in range(j):
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

#print(two_body_op(X,0,3))
#qmc_term = tensor_prod(I,I) - tensor_prod(X,X) - tensor_prod(Y,Y) - tensor_prod(Z,Z)

triangle_bd = 0
adj_mat = [[0 for k in range(n)] for l in range(n)] 
for j in range(n):
    for i in range(j):
        #entry = 1 if i == 0 else 0 # for star graphs, the maximum should always be 2n. the minimum is 0 because repetition code is in the ground space. 
        entry = np.random.rand()
        #entry = 1 if j-i==1 else 0
        adj_mat[i][j] = entry
        triangle_bd += 4*entry

adj_mat = symmetrize_mat(adj_mat)
adj_mat = np.array(adj_mat)

H = None
for j in range(n):
    for i in range(j):
        if H is None:
            H = qmc_term(adj_mat[i][j], i, j)
        else:
            H += qmc_term(adj_mat[i][j], i, j)

'''
sum_Z = None
for i in range(n):
    if sum_Z is None:
        sum_Z = one_body_op(Z, i)
    else:
        sum_Z += one_body_op(Z,i)


import pdb; pdb.set_trace()
'''

print(sorted(np.linalg.eigvals(H)))
print('triangle bound', triangle_bd)
