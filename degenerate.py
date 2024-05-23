import numpy as np
import matplotlib.pyplot as plt 
import os

# strong convexity parameter
mu = 1
# initial probability for beta > 0
prob0 = 0.5
# initial probability for beta = 0
prob00 = 0.1

# general linear-constrained settings

# generate the degenerate example for k = 2
d1 = 5
d2 = 2
k = 2
prob_seed = 3
np.random.seed(prob_seed)
mat_S = np.random.randn(d1, d1) / np.sqrt(d1)
mat_S = mat_S @ mat_S.T
mat_S = mat_S + mu * np.identity(d1)
# let c = S x^star - b
vec_c = np.random.randn(d1)
tmp = np.array([vec_c, mat_S @ vec_c])
[mat_A0, _, _] = np.linalg.svd(tmp.T)
# construct the constrained matrix
mat_A = mat_A0[:,:3]
# construct a vector v lie in col(A^bot), in fact, v equals to x^star
vec_v = mat_A0[:,3]
vec_b = mat_S @ vec_v - vec_c
# print(mat_A.shape)
x_star = np.linalg.solve(mat_S, vec_c + vec_b)
print(x_star, np.linalg.norm(mat_A.T @ x_star), np.linalg.norm(vec_v - x_star))

# check the solution
proj_A = mat_A @ np.linalg.inv(mat_A.T @ mat_A) @ mat_A.T
proj_A_bot = np.identity(d1) - proj_A
proj_A_bot = np.identity(d1) - proj_A
[eig_D, eig_U] = np.linalg.eigh(proj_A_bot)

U1 = eig_U[:,eig_D < 1e-8]
U2 = eig_U[:,eig_D > 1e-8]
sol = U2 @ np.linalg.inv(U2.T @ mat_S @ U2) @ U2.T @ vec_b

print(np.linalg.norm(x_star - sol), np.linalg.norm(vec_c - mat_S @ sol + vec_b))
print(np.linalg.norm(proj_A_bot @ vec_c), np.linalg.norm(proj_A_bot @ mat_S @ vec_c),\
     np.linalg.norm(proj_A_bot @ mat_S @ mat_S @ vec_c))
print(np.linalg.norm(proj_A_bot @ mat_S @ mat_S @ mat_S @ vec_c))

# run
alpha_list = [1.0, 0.8]
beta_list = [0.0, 0.4, 0.83, 0.87, 0.9, 0.95]

eta0 = 1
eta00 = 0.2
var = 1
n_ite = 100000
prob00 = 0.1
n_rep = 10

u_rep_list = []
v_rep_list = []
mse_rep_list = []
loss_rep_list = []
# x0 = np.random.randn(d1)

store_path = '../workspace/LPSA/dege' + str(k) + '_mse_seed' + str(prob_seed) + '/'

for alpha in alpha_list:
    for beta in beta_list:
        para = 'alpha' + str(alpha) + 'beta' + str(beta)
        path = store_path + para + '/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        if alpha > 0.99:
            eta_ini = eta0
        else:
            eta_ini = eta00
        print(eta_ini)

        for j in range(n_rep):
#             np.random.seed(j)
            x0 = np.random.randn(d1)
            u_list = []
            v_list = []
            mse_list = []
            loss_list = []
            eta_list = []
            f_list = []
            x = x0
            for i in range(n_ite):
                eta = eta_ini / (i+1) ** alpha
                noise = np.sqrt(var) * np.random.randn(d1)
                x = x - eta * ((mat_S @ x - vec_b) + noise)
                u = proj_A_bot @ x

                if beta > 0.01:
                    prob = prob0 * eta ** beta
                else:
                    prob = prob00
                prob = min(prob, 1)
                f = np.random.binomial(1, prob)

                if f > 0:
                    x = u
                v = x - u
                u_list.append(u - sol)
                v_list.append(v)
                mse = np.linalg.norm(u - sol) ** 2
#                 loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                mse_list.append(mse)
#                 loss_list.append(loss - loss_min)
                eta_list.append(eta)
                f_list.append(f)
#             print(np.sum(np.array(f_list)))
            np.save(path + 'mse' + str(j) + '.npy', np.array(mse_list))
#             np.save(path + 'loss' + str(j) + '.npy', np.array(loss_list))
#             np.save(path + 'eta' + str(j) + '.npy', np.array(eta_list))
#             np.save(path + 'u' + str(j) + '.npy', np.array(u_list))
#             np.save(path + 'v' + str(j) + '.npy', np.array(v_list))
            np.save(path + 'f' + str(j) + '.npy', np.array(f_list))



# generate the degenerate example for k = 3
d1 = 20
d2 = 10
k = 3
prob_seed = 1
np.random.seed(prob_seed)
mat_S = np.random.randn(d1, d1) / np.sqrt(d1)
mat_S = mat_S @ mat_S.T
mat_S = mat_S + mu * np.identity(d1)
# let c = S x^star - b
vec_c = np.random.randn(d1)
tmp = np.array([vec_c, mat_S @ vec_c, mat_S @ mat_S @ vec_c])
[mat_A0, _, _] = np.linalg.svd(tmp.T)
# construct the constrained matrix
mat_A = mat_A0[:,:3]
# construct a vector v lie in col(A^bot), in fact, v equals to x^star
vec_v = mat_A0[:,3]
vec_b = mat_S @ vec_v - vec_c
# print(mat_A.shape)
x_star = np.linalg.solve(mat_S, vec_c + vec_b)
print(x_star, mat_A.T @ x_star)

# check the solution
proj_A = mat_A @ np.linalg.inv(mat_A.T @ mat_A) @ mat_A.T
proj_A_bot = np.identity(d1) - proj_A
[eig_D, eig_U] = np.linalg.eigh(proj_A_bot)

U1 = eig_U[:,eig_D < 1e-8]
U2 = eig_U[:,eig_D > 1e-8]
sol = U2 @ np.linalg.inv(U2.T @ mat_S @ U2) @ U2.T @ vec_b

print(np.linalg.norm(x_star - sol), np.linalg.norm(vec_c - mat_S @ sol + vec_b))
print(np.linalg.norm(proj_A_bot @ vec_c), np.linalg.norm(proj_A_bot @ mat_S @ vec_c), \
      np.linalg.norm(proj_A_bot @ mat_S @ mat_S @ vec_c))
print(np.linalg.norm(proj_A_bot @ mat_S @ mat_S @ mat_S @ vec_c))

# run

alpha_list = [1.0, 0.8]
beta_list = [0.0, 0.4, 0.83, 0.87, 0.9, 0.95]

eta0 = 1
eta00 = 0.2
var = 1
n_ite = 100000
prob00 = 0.1
n_rep = 10

u_rep_list = []
v_rep_list = []
mse_rep_list = []
loss_rep_list = []
# x0 = np.random.randn(d1)

store_path = '../workspace/LPSA/dege' + str(k) + '_mse_seed' + str(prob_seed) + '/'

for alpha in alpha_list:
    for beta in beta_list:
        para = 'alpha' + str(alpha) + 'beta' + str(beta)
        path = store_path + para + '/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        if alpha > 0.99:
            eta_ini = eta0
        else:
            eta_ini = eta00
        print(eta_ini)

        for j in range(n_rep):
#             np.random.seed(j)
            x0 = np.random.randn(d1)
            u_list = []
            v_list = []
            mse_list = []
            loss_list = []
            eta_list = []
            f_list = []
            x = x0
            for i in range(n_ite):
                eta = eta_ini / (i+1) ** alpha
                noise = np.sqrt(var) * np.random.randn(d1)
                x = x - eta * ((mat_S @ x - vec_b) + noise)
                u = proj_A_bot @ x

                if beta > 0.01:
                    prob = prob0 * eta ** beta
                else:
                    prob = prob00
                prob = min(prob, 1)
                f = np.random.binomial(1, prob)

                if f > 0:
                    x = u
                v = x - u
                u_list.append(u - sol)
                v_list.append(v)
                mse = np.linalg.norm(u - sol) ** 2
#                 loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                mse_list.append(mse)
#                 loss_list.append(loss - loss_min)
                eta_list.append(eta)
                f_list.append(f)
#             print(np.sum(np.array(f_list)))
            np.save(path + 'mse' + str(j) + '.npy', np.array(mse_list))
#             np.save(path + 'loss' + str(j) + '.npy', np.array(loss_list))
#             np.save(path + 'eta' + str(j) + '.npy', np.array(eta_list))
#             np.save(path + 'u' + str(j) + '.npy', np.array(u_list))
#             np.save(path + 'v' + str(j) + '.npy', np.array(v_list))
            np.save(path + 'f' + str(j) + '.npy', np.array(f_list))


# FL settings

# generate the vector b in a FL sample
# ss: the diagonal values of the matrix S
# k: the parameter degeneration
# dim: the dimension of the matrix S
def gen_b(ss, k, dim, n_client, prob_seed = 1):
    np.random.seed(prob_seed)
    if not isinstance(k, int) or k < 1 or k > n_client:
        raise ValueError('Wrong value of k')
    if ss.shape[0] != n_client or ss.shape[1] != dim:
        raise ValueError('Wrong shape of ss')
    
    coef_vec = np.zeros(n_client)
    b_list = []
    for j in range(dim):
        coef_mat = []
        for i in range(1, k+1):
            s = ss[:, j]
            coef_mat.append( np.sum(s ** (i+1)).repeat(n_client) - np.sum(s) * s ** i)
        for i in range(k+1, n_client+1):
            coef_mat.append(np.random.randn(n_client))
        coef_mat = np.array(coef_mat)
        coef_vec[k-1] = 5
        b_list.append( np.linalg.solve(coef_mat, coef_vec) )
    return np.array(b_list)

# generate the degenerate exapmle for k = 2
k = 2
dim = 2
n_client = 5
prob_seed = 2
s = np.random.randn(n_client, dim) ** 2 + 1
vec_b = gen_b(s, k, dim, n_client, prob_seed).T

# sol = np.sum(b) / np.sum(s)
print(vec_b.shape)

mat_S_list = []
for i in range(n_client):
    mat_S_list.append(np.diag(s[i,:]))
mat_S = np.array(mat_S_list)

mat_S_ave = np.mean(mat_S, axis=0)
vec_b_ave = np.mean(vec_b, axis=0)
sol = np.linalg.inv(mat_S_ave) @ vec_b_ave
print(sol)

# run
alpha_list = [1.0, 0.8]
beta_list = [0.0, 0.4, 0.75, 0.8, 0.85, 0.9, 0.95]
eta0 = 1
eta00 = 0.2

prob00 = 0.1 # initial value for beta = 0
prob0 = 0.5 # initial value for beta > 0
n_ite = 100000
n_rep = 10

store_path = '../workspace/LPSA/fl_dege' + str(k) + '_mse_seed' + str(prob_seed) + '/'


for alpha in alpha_list:
    for beta in beta_list:
        para = 'alpha' + str(alpha) + 'beta' + str(beta)
        path = store_path + para + '/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        if alpha > 0.99:
            eta_ini = eta0
        else:
            eta_ini = eta00
        # print(eta_ini)

        for j in range(n_rep):
#             np.random.seed(j)
            x0 = np.random.randn(n_client, dim)
#             x0 = np.random.randn(n_client)
            u_list = []
            v_list = []
            mse_list = []
            loss_list = []
            eta_list = []
            f_list = []
            x = x0
            for i in range(n_ite):
                eta = eta_ini / (i+1) ** alpha
                noise = np.sqrt(var) * np.random.randn(n_client, dim)
#                 print(mat_S.shape, x.shape)
                grad = np.sum(mat_S * np.expand_dims(x, axis=1).repeat(dim, axis=1), axis=2) - vec_b
#                 grad = s * x - b
                x = x - eta * (grad + noise)
                u = np.mean(x, axis=0)

                if beta > 0.01:
                    prob = prob0 * eta ** beta
                else:
                    prob = prob00
                prob = min(prob, 1)
                f = np.random.binomial(1, prob)

                if f > 0:
                    x = np.repeat(np.expand_dims(u, axis=0), 5, axis=0)
#                     x = np.repeat(u, n_client)
                v = x - u
                u_list.append(u - sol)
                v_list.append(v)
                mse = np.linalg.norm(u - sol) ** 2
#                 loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                mse_list.append(mse)
#                 loss_list.append(loss - loss_min)
                eta_list.append(eta)
                f_list.append(f)

            np.save(path + 'mse' + str(j) + '.npy', np.array(mse_list))
#             np.save(path + 'loss' + str(j) + '.npy', np.array(loss_list))
#             np.save(path + 'eta' + str(j) + '.npy', np.array(eta_list))
#             np.save(path + 'u' + str(j) + '.npy', np.array(u_list))
#             np.save(path + 'v' + str(j) + '.npy', np.array(v_list))
            np.save(path + 'f' + str(j) + '.npy', np.array(f_list))

        # print(para, j, 'done\n')


# generate the degenerate example for k = 3
k = 3
dim = 4
n_client = 5
prob_seed = 1
s = np.random.randn(n_client, dim) ** 2 + 1
vec_b = gen_b(s, k, dim, n_client, prob_seed).T

# sol = np.sum(b) / np.sum(s)
print(vec_b.shape)

mat_S_list = []
for i in range(n_client):
    mat_S_list.append(np.diag(s[i,:]))
mat_S = np.array(mat_S_list)

mat_S_ave = np.mean(mat_S, axis=0)
vec_b_ave = np.mean(vec_b, axis=0)
sol = np.linalg.inv(mat_S_ave) @ vec_b_ave
print(sol)

# run
alpha_list = [1.0, 0.8]
beta_list = [0.0, 0.4, 0.83, 0.87, 0.9, 0.95]

# seed = 1
eta0 = 1
eta00 = 0.3

prob00 = 0.1 # initial value for beta = 0
prob0 = 0.5 # initial value for beta > 0
n_ite = 100000
n_rep = 10

store_path = '../workspace/LPSA/fl_dege' + str(k) + '_mse_seed' + str(prob_seed) + '/'

# np.random.seed(0)
# x00 = np.random.randn(n_client, dim)

for alpha in alpha_list:
    for beta in beta_list:
        para = 'alpha' + str(alpha) + 'beta' + str(beta)
        path = store_path + para + '/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        if alpha > 0.99:
            eta_ini = eta0
        else:
            eta_ini = eta00
        # print(eta_ini)

        for j in range(n_rep):
#             np.random.seed(j)
            x0 = np.random.randn(n_client, dim)
#             x0 = np.random.randn(n_client)
            u_list = []
            v_list = []
            mse_list = []
            loss_list = []
            eta_list = []
            f_list = []
            x = x0
            for i in range(n_ite):
                eta = eta_ini / (i+1) ** alpha
                noise = np.sqrt(var) * np.random.randn(n_client, dim)
#                 print(mat_S.shape, x.shape)
                grad = np.sum(mat_S * np.expand_dims(x, axis=1).repeat(dim, axis=1), axis=2) - vec_b
#                 grad = s * x - b
                x = x - eta * (grad + noise)
                u = np.mean(x, axis=0)

                if beta > 0.01:
                    prob = prob0 * eta ** beta
                else:
                    prob = prob00
                prob = min(prob, 1)
                f = np.random.binomial(1, prob)

                if f > 0:
                    x = np.repeat(np.expand_dims(u, axis=0), 5, axis=0)
#                     x = np.repeat(u, n_client)
                v = x - u
                u_list.append(u - sol)
                v_list.append(v)
                mse = np.linalg.norm(u - sol) ** 2
#                 loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                mse_list.append(mse)
#                 loss_list.append(loss - loss_min)
                eta_list.append(eta)
                f_list.append(f)

            np.save(path + 'mse' + str(j) + '.npy', np.array(mse_list))
#             np.save(path + 'loss' + str(j) + '.npy', np.array(loss_list))
#             np.save(path + 'eta' + str(j) + '.npy', np.array(eta_list))
#             np.save(path + 'u' + str(j) + '.npy', np.array(u_list))
#             np.save(path + 'v' + str(j) + '.npy', np.array(v_list))
#             np.save(path + 'f' + str(j) + '.npy', np.array(f_list))

        # print(para, j, 'done\n')