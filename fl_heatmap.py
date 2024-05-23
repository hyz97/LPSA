import numpy as np
import matplotlib.pyplot as plt 
import os

alpha = 1
beta = 0
dim = 2
n_client = 5
mu = 1
var = 1

# generate the problem
prob_seed = 7
np.random.seed(prob_seed)
mat_S_list = []
vec_b_list = []
for i in range(n_client):
    mat_tmp = np.random.randn(dim, dim) / np.sqrt(dim)
    mat_tmp = mat_tmp @ mat_tmp.T
    mat_tmp = mat_tmp + mu * np.identity(dim)
    mat_S_list.append(mat_tmp)
    vec_b_list.append(np.random.randn(dim))
mat_S = np.array(mat_S_list)
vec_b = np.array(vec_b_list)
print(mat_S.shape)
print(vec_b.shape)
    
# mat_A = np.random.randn(d1, d1 - d2)
# proj_A = mat_A @ np.linalg.inv(mat_A.T @ mat_A) @ mat_A.T
# proj_A_bot = np.identity(d1) - proj_A
# print(proj_A_bot @ proj_A_bot - proj_A_bot)

# solve the problem
mat_S_ave = np.mean(mat_S, axis=0)
vec_b_ave = np.mean(vec_b, axis=0)
sol = np.linalg.inv(mat_S_ave) @ vec_b_ave
print(sol)


# np.random.seed(0)

# alpha_list = [1.0, 0.8]
alpha_list = [1.0, 0.8]
# beta_list = [0.0, 0.2, 0.5]
# beta_list = [0.5]
beta_list = [0.0, 0.2]

# seed = 0
# eta0 = 1 # initial value for alpha = 1
# # eta0 = 0.5
# eta00 = 0.1 # initial value for alpha < 1

# seed = 1
eta0 = 1
eta00 = 0.2

prob00 = 0.1 # initial value for beta = 0
prob0 = 0.5 # initial value for beta > 0
n_ite = 10000
n_rep = 10000


store_path = '../workspace/LPSA/fl_heatmap_seed' + str(prob_seed) + '/'
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

        u_list = []

        for j in range(n_rep):
            # np.random.seed(j)
            x0 = np.random.randn(n_client, dim)
            
            # v_list = []
            # mse_list = []
            # loss_list = []
            # eta_list = []
            # f_list = []
            x = x0
            for i in range(n_ite):
                eta = eta_ini / (i+1) ** alpha
                noise = np.sqrt(var) * np.random.randn(n_client, dim)
#                 print(mat_S.shape, x.shape)
                grad = np.sum(mat_S * np.expand_dims(x, axis=1).repeat(dim, axis=1), axis=2) - vec_b
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
                v = x - u
                # u_list.append(u - sol)
                # v_list.append(v)
                # mse = np.linalg.norm(u - sol) ** 2
#                 loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                # mse_list.append(mse)
#                 loss_list.append(loss - loss_min)
                # eta_list.append(eta)
                # f_list.append(f)

            u_list.append((u - sol) / np.sqrt(eta))
                # v_list.append(v)
                # mse = np.linalg.norm(u - sol) ** 2
                # loss = np.dot(mat_S @ u, u) / 2 - np.dot(u, vec_b)
                # mse_list.append(mse)
                # loss_list.append(loss - loss_min)
                # eta_list.append(eta)
                # f_list.append(f)
            
            if j % 1000 == 0:
                print('rep', j)

            
            # np.save(path + 'mse' + str(j) + '.npy', np.array(mse_list))
            # np.save(path + 'loss' + str(j) + '.npy', np.array(loss_list))
            # np.save(path + 'eta' + str(j) + '.npy', np.array(eta_list))
        np.save(path + 'u' + '.npy', np.array(u_list))
            # np.save(path + 'v' + str(j) + '.npy', np.array(v_list))
            # np.save(path + 'f' + str(j) + '.npy', np.array(f_list))
        print(para, j, 'done\n')
            