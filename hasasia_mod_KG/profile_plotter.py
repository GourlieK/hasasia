import time, os, subprocess, random
import matplotlib.pyplot as plt
import numpy  as np



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

    #grabbing initial time stamp
    data = []
    with open(psrs_name_path + '/Null_time.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            data.append(line)

    null_time = float(data[0])
    del data

                
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


  
    rrf_handles = []
    original_handles = []

    # Grabbing increment data and plotting
    for i in range(len(increms)):
        name = increms[i][0] + increms[i][1]
        val_1 = float(increms[i][2])
        val_2 = float(increms[i][3])
        index = np.where((time_data >= val_1) & (time_data <= val_2))
        x_vals = time_data[index]
        y_vals = mem_data[index]
        line, = plt.plot(x_vals, y_vals, label=name)
        if "RRF" in name:
            rrf_handles.append(line)
        else:
            original_handles.append(line)

    # Plot each pulsar data
    plt.title(f"Memory vs Time of {len(increms)} Pulsars from hasasia_spectrum")
    plt.xlabel('Time [s]')
    plt.ylabel('RAM Memory Usage [MB]')
    plt.grid()
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlim(time_data[0] - 1, time_data[-1] + 1)

    # Create first legend for RRF data
    first_legend = plt.legend(handles=rrf_handles, loc='upper left', bbox_to_anchor=(-0.18, 1.15))

    # Create second legend for original data
    second_legend = plt.legend(handles=original_handles, loc='upper right', bbox_to_anchor=(1.12, 1.15))

    # Add the legends to the plot
    plt.gca().add_artist(first_legend)
    plt.gca().add_artist(second_legend)

    # Save and show the plot
    plt.savefig(psrs_name_path+'/colored_mem_time.png')
    plt.show()



    with open(psrs_name_path + '/psr_increm.txt', 'r') as file:
        OGS = {}
        RRFS = {}
        for line in file:
            parse_line = line.split()
            if parse_line[0] == 'OG':
                OGS.update({parse_line[1]: (float(parse_line[3])-float(parse_line[2]))})

            elif parse_line[0] == 'RRF':
                RRFS.update({parse_line[1]: (float(parse_line[3])-float(parse_line[2]))})

        all_names = OGS.keys()
        ogs_plot_values = [OGS[name] if name in OGS else 0 for name in all_names]
        rrfs_plot_values = [RRFS[name] if name in RRFS else 0 for name in all_names]

        # Plotting
        x = range(len(all_names))

        plt.figure(figsize=(14, 7))

        # Bar width
        bar_width = 0.4

        # Plot OGS values
        plt.bar(x, ogs_plot_values, width=bar_width, label='Original', color='blue', align='center')

        # Plot RRFS values (shifted to the right by bar_width)
        plt.bar([i + bar_width for i in x], rrfs_plot_values, width=bar_width, label='RRF', color='red', align='center')

        # Add names to the x-axis
        plt.xticks([i + bar_width / 2 for i in x], all_names, rotation=90)

        plt.xlabel('Psrs')
        plt.ylabel('Time [s]')
        plt.title('12.5 yr dataset with Thinning of 2')
        plt.legend(loc='upper right')

        # Show plot
        plt.tight_layout()
        plt.savefig(psrs_name_path+'/time_bar.png') 
        plt.show()













