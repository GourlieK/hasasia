import time, psutil, os, signal, gc, shutil
import numpy as np
import jax.numpy as jnp
import multiprocessing as mp


#Function computing matrix multiplication by writing resultant matrix onto .npy file
def disk_matrix_mult(A,B): 
    row, col= A.shape[0], B.shape[1]
    if int(A.shape[1]) != int(B.shape[0]):                          
        print('Error. Matrices dimensions are not compactable')
        exit()
    dir_path = '/home/gourliek/Desktop/matrix_data'
    path = dir_path + '/result.npy'
    try:
        os.mkdir(dir_path)
    except:
        shutil.rmtree(dir_path)
        print('Copy of File was found. Was deleted')
        os.mkdir(dir_path)
    file = np.memmap(path, dtype='d', mode='w+', shape=(A.shape[0],B.shape[1]))
    for i in range(A.shape[0]):
        file[i,:] = np.dot(A[i], B)
        file.flush()
        print(f'\t{round(i/A.shape[0] * 100,3)}% complete')
    conn2.send('DISK') 
    queue_disk_matrix.put(f'[{row},{col},{path}]')
    print('Disk Matrix Multiplication Complete')
    

#This function constantly checks the virtual RAM of the machine
def RAM_checker():
    safe_boundary = 73
    counter = 0
    while True:
        #find ways to write already computed matrix multiplcation values to disk to save time
        #add future conditions to include both types of matrix multiplication
        #if counter is 1, writes Spectra to Disk
        #if counter is 2, computes matrix multiplication by writing resultant to disk
        #if counter is 3, computes matrix multiplication by writing the two matrices that
        #you are trying to find the product of to disk
        if counter == 0:
            if float(psutil.virtual_memory().percent) >= safe_boundary:
                print(f'Memory Usage: {psutil.virtual_memory().percent}%')
                print(f'RAM usage is too high!')
                print('Switching Forms of Matrix Multiplication')
                os.kill(PID_2,signal.SIGKILL)    
                time.sleep(1)
                p_3 = mp.Process(target=disk_matrix_mult,args = (A,B))
                p_3.start()
                counter += 1
                
            else:
                print('\rMemory Usage: {0}%'.format(psutil.virtual_memory().percent),end='')
        else:
            print('\rMemory Usage: {0}%'.format(psutil.virtual_memory().percent),end='')
            

#computes matrix multiplication using jax.numpy and using Memory
def matrix_mult(A,B):
    C = jnp.matmul(A,B)
    queue_mem_matrix.put(C)
    print('Memory Matrix Mult. Complete')
    conn2.send('MEM')     #important that this is last that way signal isn't sent too early
        
        

if __name__ == "__main__":
    print(f'Initial RAM Usage: {psutil.virtual_memory().percent}%')
    queue_mem_matrix = mp.Queue()     #creates a queue to send memory matrix product to
    queue_disk_matrix = mp.Queue()    #creates a queue to send disk matrix product to
    conn1, conn2 = mp.Pipe()          #creates a pipe to send the decision whether or not memory or disk is being used
    mm =15000                         #random dimensions of the randomly generated matrices
    A = np.random.rand(mm, mm)        #Matrix A
    B = np.random.rand(mm, mm)        #Matrix B
    p_1 = mp.Process(target=RAM_checker,args = ())  #creating multiprocess of the Ram checker
    p_2 = mp.Process(target=matrix_mult,args = (A,B))#creating multiprocess of the memory matrix multiplication function
    p_2.start()     #starting the multiprocess for matrix multiplication
    PID_2 = p_2.pid #grabing the process ID of said multiprocess
    p_1.start()     #starting the multiprocess for checking the ram
    PID_1 = p_1.pid #grabing the processs ID of said multiprocess
    response = conn1.recv()    #getting the decision on whether or not memory or ram was used to compute matrix multiplication
    A = None                  #attempting to remove matrix A from memory to save space
    B = None                  #attmepting to remove matrix B from memory to save space
    del A,B                   #deleing matrix A and B
    gc.collect()              #forcing the python garbage collector to remove matrix A and B from memory
    #these if-statements are used to figure out which subprocess that the code needs to wait on to complete
    if response == 'MEM':
        mem_product = queue_mem_matrix.get()   #here, the main code won't execute in parallel anymore unto this queue is sent
        print(f'Memory Product: \n {mem_product}')
        del mem_product
    if response == 'DISK':
        data_raw = queue_disk_matrix.get()    #here, the main code won't execute in parallel anymore unto this queue is sent
        data = data_raw.strip('][').split(',')
        row = int(data[0])
        col = int(data[1])
        path = data[2]
        result = np.memmap(path, dtype='d', mode='r', shape=(row,col))
        disk_product = np.array(result)       #this offically puts the matrix into memory
        print(f'Disk Product: \n {disk_product}')
        del disk_product
    os.kill(PID_1,signal.SIGKILL)    #once complete, kills the ram checker subprocess
    print('done')



