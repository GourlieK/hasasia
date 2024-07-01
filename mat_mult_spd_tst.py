import numpy as np
import jax.numpy as jnp
from jax import lax
import time
import matplotlib.pyplot as plt

def numpy_matmult(A, B, C):
    result = A @ B @ C

def jax_numpy_matmult(A, B, C):
    result = jnp.matmul(A, jnp.matmul(B, C))

def lax_matmult(A, B, C):
    result = lax.batch_matmul(A, lax.batch_matmul(B, C))



if __name__ == "__main__":
    dim_max = 5000
    num = 1
    dims = []
    numpy_ls = []
    numpy_jax_ls = []
    jax_lax_ls = []
    while num <= dim_max:
        A= np.random.rand(num, num)
        B= np.random.rand(num, num)
        C= np.random.rand(num, num)

        #time_in = time.time()
        #numpy_matmult(A, B, C)
        #time_out = time.time()
        #numpy_ls.append(time_out-time_in)

        time_in = time.time()
        jax_numpy_matmult(A, B, C)
        time_out = time.time()
        numpy_jax_ls.append(time_out-time_in)

        time_in = time.time()
        lax_matmult(A, B, C)
        time_out = time.time()
        jax_lax_ls.append(time_out-time_in)

        dims.append(num)
        num+=1
        amount_left = round(num/dim_max * 100, 2)
        print('\r{0}% Complete'.format(amount_left), end='', flush=True)
    
    plt.title('Time vs Dimension of 3 Matrix Multiplications')
    #plt.plot(dims, numpy_ls, label='numpy')
    plt.plot(dims, numpy_jax_ls, label='jax.numpy.matmult', c='red')
    plt.plot(dims, jax_lax_ls, label='jax.lax', c='blue')
    plt.xlabel('Dimension')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid()
    plt.savefig('mat_mult_spd.png')
    plt.show()

    

