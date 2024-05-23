import numpy as np
import matplotlib.pyplot as plt 
import os

alpha = 1
beta = 0
d1 = 5
d2 = 2
mu = 1
var = 1

# generate the problem
prob_seed = 1
np.random.seed(prob_seed)
mat_S = np.random.randn(d1, d1) / np.sqrt(d1)
mat_S = mat_S @ mat_S.T
mat_S = mat_S + mu * np.identity(d1)
vec_b = np.random.randn(d1)

# generate the constraint and projection
mat_A = np.random.randn(d1, d1 - d2)
proj_A = mat_A @ np.linalg.inv(mat_A.T @ mat_A) @ mat_A.T
proj_A_bot = np.identity(d1) - proj_A
print(proj_A_bot @ proj_A_bot - proj_A_bot)

# solve the problem
[eig_D, eig_U] = np.linalg.eigh(proj_A_bot)
U1 = eig_U[:,eig_D < 1e-8]
U2 = eig_U[:,eig_D > 1e-8]
sol = U2 @ np.linalg.inv(U2.T @ mat_S @ U2) @ U2.T @ vec_b
sol_uncons = np.linalg.inv(mat_S) @ vec_b
loss_min = np.dot(mat_S @ sol, sol) / 2 - np.dot(sol, vec_b)
# print(loss_min, -np.dot(sol_uncons, vec_b)/2 )
# print(sol, sol_uncons)


# np.random.seed(0)

alpha_list = [1.0, 0.8, 0.6]
# alpha_list = [1.0]
# alpha_list = [0.8, 0.6]
beta_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]

# seed = 0
# eta0 = 1 # initial value for alpha = 1
# # eta0 = 0.5
# eta00 = 0.1 # initial value for alpha < 1

# seed = 1
eta0 = 1
eta00 = 0.2

prob00 = 0.1 # initial value for beta = 0
prob0 = 0.5 # initial value for beta > 0
n_ite = 100000
n_rep = 10

store_path = '../workspace/LPSA/mse_seed' + str(prob_seed) + '/'
# store_path = '../workspace/LPSA/mse_eta0.5/'

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
            # if j == 0:
            #     np.random.seed(j)
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
                loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                mse_list.append(mse)
                loss_list.append(loss - loss_min)
                eta_list.append(eta)
                f_list.append(f)

            np.save(path + 'mse' + str(j) + '.npy', np.array(mse_list))
            np.save(path + 'loss' + str(j) + '.npy', np.array(loss_list))
            np.save(path + 'eta' + str(j) + '.npy', np.array(eta_list))
            np.save(path + 'u' + str(j) + '.npy', np.array(u_list))
            np.save(path + 'v' + str(j) + '.npy', np.array(v_list))
            np.save(path + 'f' + str(j) + '.npy', np.array(f_list))

            print(para, j, 'done\n')