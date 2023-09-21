import time, psutil, os, signal, h5py
import numpy as np
import multiprocessing as mp
from fast_for_loop import fast_for_loop


def disk_matrix_mult(A,B):   
    
    if int(A.shape[1]) != int(B.shape[0]):                          
        print('Error. Matrices dimensions are not compactable')
        exit()
    conn2.send('DISK')
    path = r'/home/gourliek/Desktop/matrix_data'
    try:
        os.mkdir(path)
    except os.error:
        print()

    filename_1 = path + '/matrix_1.txt'
    filename_2 = path + '/matrix_2.txt'

    fp_1 = open(filename_1, 'w+')
    fp_2 = open(filename_2, 'w+')
    result = np.empty((int(A.shape[0]),int(B.shape[1])),dtype = 'd')  
    for i in range(int(A.shape[0])):
        fp_1.write(f'{np.ndarray.tolist(A[i,:])}\n')     #places all rows of A in text file cause thats how matrix mult works
    print('Successfully Wrote Matrix to first file')
    del A
    for j in range(int(B.shape[1])):
        fp_2.write(f'{np.ndarray.tolist(B[:,j])}\n')     #places all columns of B in text file cause thats how matrix mult works
    print('Successfully Wrote Matrix to Second file')
    del B
    fp_1.close()
    fp_2.close()
    print('Computing matrix mult. via text files')
    result = fast_for_loop(filename_1,filename_2,result)
    queue_disk_matrix.put(result)
    print('Completed')
    
    
   

def RAM_checker():
    safe_boundary = 80
    while True:
        if float(psutil.virtual_memory().percent) >= safe_boundary:
            print(f'Memory Usage: {psutil.virtual_memory().percent}%')
            print(f'RAM usage is too high!')
            print('Switching Forms of Matrix Multiplication')
            os.kill(PID_2,signal.SIGKILL)    #kills normal matrix multiplication function
            time.sleep(1)
            disk_mult_process()
            
        else:
            print('\rMemory Usage: {0}%'.format(psutil.virtual_memory().percent),end='')
            

def disk_mult_process():
    p_3 = mp.Process(target=disk_matrix_mult,args = (A,B))
    p_3.start()
    

    




def matrix_mult(A,B):
        C = A @ B
        del A, B
        queue_mem_matrix.put(C)
        conn2.send('MEM')
        print('Memory Matrix Mult. Complete')
        
        
if __name__ == "__main__":
    print(f'Initial RAM Usage: {psutil.virtual_memory().percent}%')
    queue_disk_matrix = mp.Queue()
    queue_mem_matrix = mp.Queue()
    conn1, conn2 = mp.Pipe()
    mm = 100
    A = np.random.rand(mm, mm)
    B = np.random.rand(mm, mm)
    p_1 = mp.Process(target=RAM_checker,args = ())
    p_2 = mp.Process(target=matrix_mult,args = (A,B))
    p_2.start()
    PID_2 = p_2.pid   
    p_1.start()    
    PID_1 = p_1.pid
    response = conn1.recv()   

    if response == 'MEM':
        mem_product = queue_mem_matrix.get()
        print(f'Memory Product: \n {mem_product}')
        del mem_product
    if response == 'DISK':
        disk_product = queue_disk_matrix.get()
        print(f'Disk Product: \n {disk_product}')
        del disk_product

    os.kill(PID_1,signal.SIGKILL)
    print()
    print('done')



