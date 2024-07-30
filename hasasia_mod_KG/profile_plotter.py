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
    psrs_name_path = os.path.expanduser('~/Desktop/Profile_Data')
    increms_psrs = []
    increms_specs = []
    psrs = []
    time_data = []
    mem_data = []

    #if data is already saved to folder, comment save_data() out and it will just analyze the data
    save_data()

    #grabbing time increments from execution of hasasia_spectrum()
    with open(psrs_name_path + '/psr_increm.txt', 'r') as file:
        for line in file:
            line = line.strip('\n').split()
            increms_psrs.append(line)

    with open(psrs_name_path + '/specs_increm.txt', 'r') as file:
        for line in file:
            line = line.strip('\n').split()
            increms_specs.append(line)

    # Grabbing initial timestamp
    with open(psrs_name_path + '/Null_time.txt', 'r') as file:
        null_time = float(file.readline().strip('\n'))

    # Read time and memory data
    with open(psrs_name_path + '/time_mem_data.txt', 'r') as file:
        for line in file:
            line = line.strip('\n').split()
            time_data.append(float(line[0]))
            mem_data.append(float(line[1]))

    # Convert to numpy arrays
    time_data = np.array(time_data)
    mem_data = np.array(mem_data)

    # Generate random colors
    hexadecimal_alphabets = '0123456789ABCDEF'
    num_colors = len(increms_psrs)
    color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for _ in range(6)]) for _ in range(num_colors)]

    # Plot generic data for 8 pulsars
    plt.plot(time_data, mem_data, c='black')
    plt.title(f"Memory vs Time of {len(increms_psrs)} Pulsars")
    plt.xlabel('time [s]')
    plt.ylabel('RAM memory usage [MB]')
    plt.grid()
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlim(time_data[0]-1, time_data[-1]+1)
    plt.savefig(psrs_name_path+'/mem_time.png')
    plt.show()

    rrf_handles_psr = []
    original_handles_psr = []
    rrf_handles_specs = []
    original_handles_specs = []

    # Grabbing increment data and plotting
    for i in range(len(increms_psrs)):
        name_psr = increms_psrs[i][0] + increms_psrs[i][1]
        val_1_psr = float(increms_psrs[i][2])
        val_2_psr = float(increms_psrs[i][3])
        
        
        index_psr = np.where((time_data >= val_1_psr) & (time_data <= val_2_psr))
        

        # Plot pulsar data
        line_psr, = plt.plot(time_data[index_psr], mem_data[index_psr], label=name_psr, color=color[i])
        if "RRF" in name_psr:
            rrf_handles_psr.append(line_psr)
        else:
            original_handles_psr.append(line_psr)
            
        

    # Plot pulsar data
    plt.title(f"Memory vs Time of {len(increms_psrs)} Pulsars from Pulsar Objects")
    plt.xlabel('Time [s]')
    plt.ylabel('RAM Memory Usage [MB]')
    plt.grid()
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlim(time_data[0] - 1, time_data[-1] + 1)

    # Create legends
    first_legend_psr = plt.legend(handles=original_handles_psr, loc='upper left', bbox_to_anchor=(-0.18, 1.15))
    second_legend_psr = plt.legend(handles=rrf_handles_psr, loc='upper right', bbox_to_anchor=(1.12, 1.15))

    # Add the legends to the plot
    plt.gca().add_artist(first_legend_psr)
    plt.gca().add_artist(second_legend_psr)

    # Save and show the plot
    plt.savefig(psrs_name_path+'/colored_mem_time_psrs.png')
    plt.show()
    

    for i in range(len(increms_specs)): 
        name_psr = increms_specs[i][0] + increms_specs[i][1]
        val_1_spec = float(increms_specs[i][2])
        val_2_spec = float(increms_specs[i][3])

        index_specs = np.where((time_data >= val_1_spec) & (time_data <= val_2_spec))

        # Plot spectrum data
        line_specs, = plt.plot(time_data[index_specs], mem_data[index_specs], label=name_psr, color=color[i])
        if "RRF" in name_psr:
            rrf_handles_specs.append(line_specs)
        else:
            original_handles_specs.append(line_specs)


    # Plot spectrum data
    plt.title(f"Memory vs Time of {len(increms_specs)} Pulsars from Spectrum Objects")
    plt.xlabel('Time [s]')
    plt.ylabel('RAM Memory Usage [MB]')
    plt.grid()
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlim(time_data[0] - 1, time_data[-1] + 1)

    # Create legends
    first_legend_specs = plt.legend(handles=original_handles_specs, loc='upper left', bbox_to_anchor=(-0.18, 1.15))
    second_legend_specs = plt.legend(handles=rrf_handles_specs, loc='upper right', bbox_to_anchor=(1.12, 1.15))

    # Add the legends to the plot
    plt.gca().add_artist(first_legend_specs)
    plt.gca().add_artist(second_legend_specs)

    # Save and show the plot
    plt.savefig(psrs_name_path+'/colored_mem_time_specs.png')
    plt.show()
    plt.close()


    #bar chart for hasasia pulsar time
    with open(psrs_name_path + '/psr_increm.txt', 'r') as file:
        OGS_psrs = {}
        RRFS_psrs = {}
        for line in file:
            parse_line = line.split()
            if parse_line[0] == 'OG':
                OGS_psrs.update({parse_line[1]: (float(parse_line[3])-float(parse_line[2]))})

            elif parse_line[0] == 'RRF':
                RRFS_psrs.update({parse_line[1]: (float(parse_line[3])-float(parse_line[2]))})

        all_names = OGS_psrs.keys()
        ogs_plot_values_psrs = [OGS_psrs[name] if name in OGS_psrs else 0 for name in all_names]
        rrfs_plot_values_psrs = [RRFS_psrs[name] if name in RRFS_psrs else 0 for name in all_names]

        # Plotting
        x = range(len(all_names))

        plt.figure(figsize=(14, 7))

        # Bar width
        bar_width = 0.4

        # Plot OGS values
        plt.bar(x, ogs_plot_values_psrs, width=bar_width, label='Original', color='blue', align='center')

        # Plot RRFS values (shifted to the right by bar_width)
        plt.bar([i + bar_width for i in x], rrfs_plot_values_psrs, width=bar_width, label='RRF', color='red', align='center')

        # Add names to the x-axis
        plt.xticks([i + bar_width / 2 for i in x], all_names, rotation=90)

        plt.xlabel('Psrs')
        plt.ylabel('Time [s]')
        plt.title('Time to Create Pulsar Objects')
        plt.legend(loc='upper right')

        # Show plot
        plt.tight_layout()
        plt.savefig(psrs_name_path+'/time_bar_psrs.png') 
        plt.show()


        #bar chart for hasasia spectrum time
        with open(psrs_name_path + '/specs_increm.txt', 'r') as file:
            OGS_specs = {}
            RRFS_specs = {}
            for line in file:
                parse_line = line.split()
                if parse_line[0] == 'OG':
                    OGS_specs.update({parse_line[1]: (float(parse_line[3])-float(parse_line[2]))})

                elif parse_line[0] == 'RRF':
                    RRFS_specs.update({parse_line[1]: (float(parse_line[3])-float(parse_line[2]))})

            all_names = OGS_specs.keys()
            ogs_plot_values_specs = [OGS_specs[name] if name in OGS_specs else 0 for name in all_names]
            rrfs_plot_values_specs = [RRFS_specs[name] if name in RRFS_specs else 0 for name in all_names]

            # Plotting
            x = range(len(all_names))

            plt.figure(figsize=(14, 7))

            # Bar width
            bar_width = 0.4

            # Plot OGS values
            plt.bar(x, ogs_plot_values_specs, width=bar_width, label='Original', color='blue', align='center')

            # Plot RRFS values (shifted to the right by bar_width)
            plt.bar([i + bar_width for i in x], rrfs_plot_values_specs, width=bar_width, label='RRF', color='red', align='center')

            # Add names to the x-axis
            plt.xticks([i + bar_width / 2 for i in x], all_names, rotation=90)

            plt.xlabel('Psrs')
            plt.ylabel('Time [s]')
            plt.title('Time to Create Spectrum Objects')
            plt.legend(loc='upper right')

            # Show plot
            plt.tight_layout()
            plt.savefig(psrs_name_path+'/time_bar_specs.png') 
            plt.show()


        #bar chart for total time
        og_psrs_tm = np.array(ogs_plot_values_psrs)
        rrf_psrs_tm = np.array(rrfs_plot_values_psrs)

        og_specs_tm = np.array(ogs_plot_values_specs)
        rrf_specs_tm = np.array(rrfs_plot_values_specs)

        total_og_tm = og_psrs_tm + og_specs_tm
        total_rrf_tm = rrf_psrs_tm + rrf_specs_tm
        x = range(len(all_names))

        plt.figure(figsize=(14, 7))
        bar_width = 0.4
        plt.bar(x, total_og_tm, width=bar_width, label='Original', color='blue', align='center')
        plt.bar([i + bar_width for i in x], total_rrf_tm, width=bar_width, label='RRF', color='red', align='center')

        # Add names to the x-axis
        plt.xticks([i + bar_width / 2 for i in x], all_names, rotation=90)

        plt.xlabel('Psrs')
        plt.ylabel('Time [s]')
        plt.title('Total Time')
        plt.legend(loc='upper right')

        # Show plot
        plt.tight_layout()
        plt.savefig(psrs_name_path+'/time_bar_total.png') 
        plt.show()















