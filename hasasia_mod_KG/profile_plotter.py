import time, os, subprocess, random
import matplotlib.pyplot as plt
import numpy  as np
from matplotlib.legend_handler import HandlerTuple



def save_data():
    dat_list = []
    #tells terminal to do a profile memory analysis of the profile_test module through the terminal
    command_1 = ["mprof", "run", "profile_test.py"]
    #lists the names of all .dat files recently created in order to grab names of each file
    command_2 = ['mprof', 'list']
    #deletes all .dat files created
    command_3 = ["mprof", "clean"]

    #"cd DIRECTORY" For some reason, os.chdir works but not subprocess.run. Has to do with permissions
    os.chdir(directory)
    subprocess.run(command_1, check = True)
    subprocess.run(command_2, check = True)
    #grabs path of .dat files
    mprof_list_path = subprocess.check_output(command_2, text=True)

    mprof_list = mprof_list_path.split('\n')
    #removes date of creation. 
    del mprof_list[-1]

    for file in mprof_list:
        new_file=file.split()
        #getting actual name of the file
        dat_list.append(new_file[1])

    #for-loop used just in case multiple .dat files were created
    for dat in dat_list:
        dat_file = os.path.join(directory,dat)
        mem = []
        times = []
        #opening .dat file
        with open(dat_file, 'r') as file:
            text = file.read()
            text = text.split('\n')
            del text[0]
            del text[-1]
            for words in text:
                raw_data = words.split(' ')
                del raw_data[0]
                mem.append(float(raw_data[0]))
                times.append(float(raw_data[1]))

        min_time = times[0]
        #setting the time to start at 0 seconds
        new_times = [Time - min_time for Time in times]
        #no longer need times list
        del times
        #saving .dat files to text file
        with open(psrs_name_path + '/time_mem_data.txt', 'w') as file:
            for i in range(len(mem)):
                file.write(f"{new_times[i]} {mem[i]}\n")

    #deleing .dat files
    subprocess.run(command_3, check = True)





if __name__ == '__main__':
    directory  = r'/home/gourliek/hasasia_clone/hasasia_mod_KG'
    psrs_name_path = r'/home/gourliek/Desktop/Profile_Data'
    increms = []
    psrs = []
    time_data = []
    mem_data = []

    #if data is already saved to folder, comment save_data() out and it will just analyze the data
    save_data()

    #grabbing time increments from execution of hasasia_spectrum()
    with open(psrs_name_path + '/psr_increm.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            line = line.split()
            increms.append(line)



    data = []
    OG = []
    OG_jax = []
    OG_mod = []
    with open(psrs_name_path + '/Ncal_meth_time.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            data.append(line)

    null_time = float(data[0])
    del data[0]

    for line in data:
        line = line.split(':')
        if str(line[0]) == 'Origonal':
            num = line[1]
            nums = num.split(',')
            new_0 = nums[0].replace("(", "")
            new_1 = nums[1].replace(")", "")
            OG.append([float(new_0),float(new_1)])

        if str(line[0]) == 'Origonal Jax':
            num = line[1]
            nums = num.split(',')
            new_0 = nums[0].replace("(", "")
            new_1 = nums[1].replace(")", "")
            OG_jax.append([float(new_0),float(new_1)])

        if str(line[0]) == 'Modified':
            num = line[1]
            nums = num.split(',')
            new_0 = nums[0].replace("(", "")
            new_1 = nums[1].replace(")", "")
            OG_mod.append([float(new_0),float(new_1)]) 
                  
    #opening the recently saved data text file
    with open(psrs_name_path + '/time_mem_data.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            line = line.split()
            time_data.append(float(line[0]))
            mem_data.append(float(line[1]))

    #generates random colors based on number of pulsars generated
    hexadecimal_alphabets = '0123456789ABCDEF'
    num_colors = len(increms)
    color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in
    range(6)]) for i in range(num_colors)]

    #made into numpy.arrays so arrays can be filtered out for each pulsar
    time_data = np.array(time_data)
    mem_data = np.array(mem_data)


    #plots generic 8 pulsars
    plt.plot(time_data,mem_data, c = 'black')
    plt.title(f"Memory vs Time of {len(increms)} Pulsars")
    plt.xlabel('time [s]')
    plt.ylabel('RAM memory usage [MB]')
    plt.grid()
    plt.axhline(y=0, color = 'black')
    plt.axvline(x = 0, color = 'black')
    plt.xlim(time_data[0]-1, time_data[-1]+1)
    plt.savefig(psrs_name_path+'/mem_time.png') 
    plt.show()

    #grabbing increment data 
    for i in range(len(increms)):
        name = increms[i][0]
        val_1 = float(increms[i][1])
        val_2 = float(increms[i][2])
        index = np.where((time_data >= val_1) & (time_data <= val_2))
        x_vals = time_data[index]
        y_vals = mem_data[index]
        plt.plot(x_vals, y_vals, c = color[i], label = name)

    #plots each pulsar data
    plt.title(f"Memory vs Time of {len(increms)} Pulsars from hasasia_spectrum")
    plt.xlabel('time [s]')
    plt.ylabel('RAM memory usage [MB]')
    plt.grid()
    plt.legend()
    plt.axhline(y=0, color = 'black')
    plt.axvline(x = 0, color = 'black')
    plt.xlim(time_data[0]-1, time_data[-1]+1)
    plt.savefig(psrs_name_path+'/colored_mem_time.png') 
    plt.show()


    x_vals_OG = []
    y_vals_OG = []
    x_vals_OG_jax = []
    y_vals_OG_jax = []
    x_vals_OG_mod = []
    y_vals_OG_mod = []

    for i in range(len(OG)):
        OG_val_1 = OG[i][0] - null_time
        OG_val_2 = OG[i][1] - null_time
        index_OG = np.where((time_data >= OG_val_1) & (time_data <= OG_val_2))
        x_vals_OG.append(time_data[index_OG])
        y_vals_OG.append(mem_data[index_OG]) 

        OG_jax_val_1 = OG_jax[i][0] - null_time
        OG_jax_val_2 = OG_jax[i][1] - null_time
        index_OG_jax = np.where((time_data >= OG_jax_val_1) & (time_data <= OG_jax_val_2))
        x_vals_OG_jax.append(time_data[index_OG_jax])
        y_vals_OG_jax.append(mem_data[index_OG_jax]) 

        OG_mod_val_1 = OG_mod[i][0] - null_time
        OG_mod_val_2 = OG_mod[i][1] - null_time
        index_OG_mod = np.where((time_data >= OG_mod_val_1) & (time_data <= OG_mod_val_2))
        x_vals_OG_mod.append(time_data[index_OG_mod]) 
        y_vals_OG_mod.append(mem_data[index_OG_mod]) 








    
    handles = []
    labels = ['Original', 'Original with Jax', 'Modified']

    for i in range(len(x_vals_OG)):
        orig_line, = plt.plot(x_vals_OG[i], y_vals_OG[i], c='black')
        jax_line, = plt.plot(x_vals_OG_jax[i], y_vals_OG_jax[i], c='red')
        mod_line, = plt.plot(x_vals_OG_mod[i], y_vals_OG_mod[i], c='blue')
        
        handles.append(orig_line)  # Appending any of the lines for the legend

    # Create legend with proxy artists
    plt.title(f"Memory vs Time of {len(increms)} Pulsars")
    plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.grid()
    plt.axhline(y=0, color = 'black')
    plt.axvline(x = 0, color = 'black')
    plt.xlabel('time [s]')
    plt.ylabel('RAM memory usage [MB]')
    plt.xlim(time_data[0]-1, time_data[-1]+1)
    plt.savefig(psrs_name_path+'/meth_mem_time.png') 
    plt.show()








