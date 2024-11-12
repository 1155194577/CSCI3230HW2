import numpy as np
import copy
import matplotlib.pyplot as plt

d = 100
m = 10
A = np.random.normal(0, 1, (m,d))
b = np.random.normal(0, 1, m)
initial_x = np.zeros(d)

def f(x,lambdas): 
    error = A@x-b
    error_l2_norm = np.linalg.norm(error)**2  
    regularizer = lambdas * np.linalg.norm(x)**2 
    return error_l2_norm + regularizer 


def coordinate_opt(A,b,lambdas,x,t):
   x_updated = copy.deepcopy(x)
   A_i_th_row = A[:,t] ## select the i-th column 
   residual = b - A @ x_updated + A_i_th_row * x_updated[t] 
   numerator = A_i_th_row@residual ## above
   denominator = A_i_th_row@A_i_th_row + lambdas ## below
   x_updated[t] = numerator/denominator 
   return x_updated 

def ridge_regresison_solver(A,b,lambdas,x_init,T=2000): 
    x = copy.deepcopy(x_init)
    counter = 0 
    ans = []
    while (counter < T): 
        t = counter % d  
        x = coordinate_opt(A,b,lambdas,x,t)
        ans.append(f(x,lambdas))
        counter += 1
    return ans

ans1 = ridge_regresison_solver(A,b,0.1,initial_x)
ans2 = ridge_regresison_solver(A,b,1.0,initial_x)
ans3 = ridge_regresison_solver(A,b,10,initial_x)
ans4 = ridge_regresison_solver(A,b,100,initial_x)

plt.plot(ans1, label='lambda=0.1')
plt.plot(ans2, label='lambda=1.0')
plt.plot(ans3, label='lambda=10')
plt.plot(ans4, label='lambda=100')

plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence of Ridge Regression Solver')
plt.legend()
plt.savefig('ridge_regression_convergence.png')