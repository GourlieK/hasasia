import time, os, subprocess, random
import matplotlib.pyplot as plt
import numpy as np


def save_data():
    dat_list = []
   
    command_1 = ["mprof", "run", "profile_test.py"]
    command_2 = ['mprof', 'list']
    command_3 = ["mprof", "clean"]



    subprocess.run(command_1, check = True)
    subprocess.run(command_2, check = True)
    mprof_list_path = subprocess.check_output(command_2, text=True)

    mprof_list = mprof_list_path.split('\n')
    del mprof_list[-1]
    for file in mprof_list:
        new_file=file.split()
        dat_list.append(new_file[1])

    for dat in dat_list:
        dat_file = os.path.join(profile_py_path,dat)
        mem = []
        times = []
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
        new_times = [Time - min_time for Time in times ]
        del times

        with open(psrs_name_path + '/time_mem_data.txt', 'w') as file:
            for i in range(len(mem)):
                file.write(f"{new_times[i]} {mem[i]}\n")

    subprocess.run(command_3, check = True)





if __name__ == '__main__':
    profile_py_path = r'/home/gourliek/hasasia_clone/hasasia_mod_KG'
    psrs_name_path = r'/home/gourliek/Desktop/Profile_Data'
    increms = []
    psrs = []
    time_data = []
    mem_data = []
    
    save_data()

    with open(psrs_name_path + '/psr_increm.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            line = line.split()
            increms.append(line)

    

    with open(psrs_name_path + '/time_mem_data.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            line = line.split()
            time_data.append(float(line[0]))
            mem_data.append(float(line[1]))



hexadecimal_alphabets = '0123456789ABCDEF'
num_colors = len(increms)
color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in
range(6)]) for i in range(num_colors)]

time_data = np.array(time_data)
mem_data = np.array(mem_data)



plt.plot(time_data,mem_data, c = 'black')
plt.title(f"Memory vs Time of {len(increms)} Pulsars")
plt.xlabel('time [s]')
plt.ylabel('RAM memory usage [MB]')
plt.grid()
plt.legend()
plt.axhline(y=0, color = 'black')
plt.axvline(x = 0, color = 'black')
plt.xlim(time_data[0]-1, time_data[-1]+1)
plt.savefig(psrs_name_path+'/mem_time.png') 
plt.show()

for i in range(len(increms)):
    name = increms[i][0]
    val_1 = float(increms[i][1])
    val_2 = float(increms[i][2])
    index = np.where((time_data >= val_1) & (time_data <= val_2))
    x_vals = time_data[index]
    y_vals = mem_data[index]
    plt.plot(x_vals, y_vals, c = color[i], label = name)

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








