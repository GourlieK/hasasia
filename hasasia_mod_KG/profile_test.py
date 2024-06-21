#Kyle Gourlie
#8/2/2023
import shutil
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import glob, pickle, json, cProfile, pstats, os, time
import matplotlib as mpl
import healpy as hp
import astropy.units as u
import astropy.constants as c
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [5,3]
mpl.rcParams['text.usetex'] = True
from enterprise.pulsar import Pulsar as ePulsar
from memory_profiler import profile



#memory profile files
path = r'/home/gourliek/Desktop/Profile_Data'
try:
    os.mkdir(path)
except FileExistsError:
    shutil.rmtree(path)
    os.mkdir(path)
corr_matrix_mem = open(path + '/corr_matrix_mem.txt','w')
sens_mem = open(path + '/sens_mem.txt','w')
sens_mem_RRF= open(path + '/sens_mem_RRF.txt','w')
time_increments = open(path + '/psr_increm.txt','w')
Ncal_time_file = open(path + '/Ncal_meth_time.txt','w')



#using modified imports
import sensitivity as hsen
import sim as hsim
import skymap as hsky
from sensitivity import get_NcalInv_mem, corr_from_psd_mem
################################################################################################################
################################################################################################################
################################################################################################################
def get_psrname(file,name_sep='_'):
    return file.split('/')[-1].split(name_sep)[0]



def rn_corr(freqs, T_obs,Amps, Gams):
    rn_corr = np.identity(2*freqs.size)
    #iterates over only the diagonal
    for i in range(rn_corr.shape[0]):
        for j in range(i,rn_corr.shape[1]):
            f_j = j/T_obs
            rn_corr[i,j] = (Amps[i]**2)/(12*np.pi**2) * 1/T_obs * (f_j/fyr)**(-Gams[i])
    
    return rn_corr



def pulsar_class(parameters, timons):
    """Generates Enterprise Pulsar Objects based on .par and .tim files

    Returns:
        enterprise_Psrs: Enterprise Pulsar Object list
    """
    enterprise_Psrs = []
    count = 1
    
    for par,tim in zip(parameters,timons):
        if count <= kill_count:
            ePsr = ePulsar(par, tim,  ephem='DE436')
            enterprise_Psrs.append(ePsr)
            print('\rPSR {0} complete'.format(ePsr.name),end='',flush=True)
            print(f'\n{count} pulsars created')
            count +=1
        else:
            break
    return enterprise_Psrs



@profile(stream = corr_matrix_mem)
def make_corr(psr):
    """Calculates the white noise covariance matrix based on EFAC, EQUAD, and ECORR

    Args:
        psr (object): Enterprise Pulsar Object

    Returns:
        corr (array): white noise covariance matrix
    """
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas,psr.toaerrs,
                                         flags=psr.flags['f'],dt=1)
    keys = [ky for ky in noise.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl,dtype=np.float32)
    for be in backends:
        mask = np.where(psr.flags['f']==be)
        key_ef = '{0}_{1}_{2}'.format(psr.name,be,'efac')
        key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'equad')
        sigma_sqr[mask] = (noise[key_ef]**2 * (psr.toaerrs[mask]**2)
                           + (10**noise[key_eq])**2)
        mask_ec = np.where(fl==be)
        #KG jax mod.
        #mask_ec = np.where(np.atleast_1d(fl == be).nonzero())
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)] 
    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J #ISSUE HERE WHEN RUNNING J1713
    return corr


@profile(stream=sens_mem_RRF)
def rrf_array_construction(epsrs):
     #lists used for characteristic strain plots
    psrs_names = []
    h_c_list = []
    freqs_list = []
    NcalInv_rrf_list = []
    

    #iterates through each enterprise pulsar
    for ePsr in epsrs:

        #benchmark stuff
        corr_matrix_mem.write(f'Pulsar: {ePsr.name}\n')
        get_NcalInv_mem.write(f'Pulsar: {ePsr.name}\n')
        corr_from_psd_mem.write(f'Pulsar: {ePsr.name}\n')
        start_time = time.time()
      
            
        #if red noise parameters for an individual pulsar is present, add it to standard red noise
        if ePsr.name in rn_psrs.keys():
            Amp, gam = rn_psrs[ePsr.name]
       
         
            #create white noise covariance matrix from enterprise pulsar 
            wn_corr = make_corr(ePsr)[::thin,::thin]
            
            #creating hasasia pulsar tobject 
            psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                                toaerrs=ePsr.toaerrs[::thin],
                                phi=ePsr.phi,theta=ePsr.theta, 
                                N=wn_corr, designmatrix=ePsr.Mmat[::thin,:])
            
            #setting name of hasasia pulsar
            psr.name = ePsr.name
            #extra variable needed for benchmarking
            psr_name = psr.name

            #enterprise pulsar is no longer needed
            del ePsr
            print(f"Hasasia Pulsar {psr.name} created\n")
            psrs_names.append(psr.name)

            #creates spectrum hasasia pulsar to calculate characteristic straing
            spec_psr = hsen.RRF_Spectrum(psr, amp = Amp, gamma = gam, freqs=freqs)

            #hasasia pulsar no longer needed
            del psr
            #_ = spec_psr.NcalInv

            h_c_list.append(spec_psr.h_c)
            freqs_list.append(spec_psr.freqs)
            NcalInv_rrf_list.append(spec_psr.NcalInv)
            del spec_psr

            print(f"Hasasia Spectrum Pulsar {psr_name} created\n")

            #benchmark stuff
            end_time = time.time()
            time_increments.write(f" RRF{psr_name} {start_time-null_time} {end_time-null_time}\n")
            print('\rPSR {0} complete'.format(psr_name),end='',flush=True)
    
    return psrs_names, h_c_list,  freqs_list, NcalInv_rrf_list



@profile(stream=sens_mem)
def array_construction(epsrs):

    #lists used for characteristic strain plots
    psrs_names = []
    h_c_list = []
    freqs_list = []
    NcalInv_list = []
    

    #iterates through each enterprise pulsar
    for ePsr in epsrs:

        #benchmark stuff
        corr_matrix_mem.write(f'Pulsar: {ePsr.name}\n')
        get_NcalInv_mem.write(f'Pulsar: {ePsr.name}\n')
        corr_from_psd_mem.write(f'Pulsar: {ePsr.name}\n')
        start_time = time.time()
      
        #create white noise covariance matrix from enterprise pulsar 
        corr = make_corr(ePsr)[::thin,::thin]

        #building red noise powerlaw using standard amplitude and gamma
        plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)

        #if red noise parameters for an individual pulsar is present, add it to standard red noise
        if ePsr.name in rn_psrs.keys():
            Amp, gam = rn_psrs[ePsr.name]
            plaw += hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)

        #adding red noise components to the noise covariance matrix via power spectral density
        corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                    toas=ePsr.toas[::thin])   
        
        #creating hasasia pulsar tobject 
        psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                            toaerrs=ePsr.toaerrs[::thin],
                            phi=ePsr.phi,theta=ePsr.theta, 
                            N=corr, designmatrix=ePsr.Mmat[::thin,:])
        
        #setting name of hasasia pulsar
        psr.name = ePsr.name
        #extra variable needed for benchmarking
        psr_name = psr.name

        #enterprise pulsar is no longer needed
        del ePsr
        print(f"Hasasia Pulsar {psr.name} created\n")
        psrs_names.append(psr.name)

        #creates spectrum hasasia pulsar to calculate characteristic straing
        spec_psr = hsen.Spectrum(psr, freqs=freqs)

        #hasasia pulsar no longer needed
        del psr

        NcalInv_list.append(spec_psr.NcalInv)
        h_c_list.append(spec_psr.h_c)
        freqs_list.append(spec_psr.freqs)
        del spec_psr

        print(f"Hasasia Spectrum Pulsar {psr_name} created\n")

        #benchmark stuff
        end_time = time.time()
        time_increments.write(f"{psr_name} {start_time-null_time} {end_time-null_time}\n")
        print('\rPSR {0} complete'.format(psr_name),end='',flush=True)
        
    return psrs_names, h_c_list,  freqs_list, NcalInv_list



def yr_11_data():
    
    #File Paths
    pardir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/'
    timdir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/'
    noise_dir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master'
    noise_dir += '/nano11y_data/noisefiles/'
    psr_list_dir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/psrlist.txt'

    #organizes files into alphabetical order
    pars = sorted(glob.glob(pardir+'*.par'))
    tims = sorted(glob.glob(timdir+'*.tim'))
    noise_files = sorted(glob.glob(noise_dir+'*.json'))

    #saving pulsar names as a list
    with open(psr_list_dir, 'r') as psr_list_file:
        psr_list = []
        for line in psr_list_file:
            new_line = line.strip("\n")
            psr_list.append(new_line)

    #filtering par and tim files to make sure they only include names found in pulsar list
    pars = [f for f in pars if get_psrname(f) in psr_list]
    tims = [f for f in tims if get_psrname(f) in psr_list]
    noise_files = [f for f in noise_files if get_psrname(f) in psr_list]
    
    if len(pars) == len(tims) and len(tims) == len(noise_files):
        pass
    else:
        print("ERROR. Filteration of tim and par files performed incorrectly")
        exit()

    noise = {}
    for nf in noise_files:
        with open(nf,'r') as fin:
            noise.update(json.load(fin))

    rn_psrs = {'B1855+09':[10**-13.7707, 3.6081],      #
           'B1937+21':[10**-13.2393, 2.46521],
           'J0030+0451':[10**-14.0649, 4.15366],
           'J0613-0200':[10**-13.1403, 1.24571],
           'J1012+5307':[10**-12.6833, 0.975424],
           'J1643-1224':[10**-12.245, 1.32361],
           'J1713+0747':[10**-14.3746, 3.06793],
           'J1747-4036':[10**-12.2165, 1.40842],
           'J1903+0327':[10**-12.2461, 2.16108],
           'J1909-3744':[10**-13.9429, 2.38219],
           'J2145-0750':[10**-12.6893, 1.32307],
           }
    return psr_list, pars, tims, noise, rn_psrs



def yr_12_data():
    data_dir = r'/home/gourliek/Nanograv/12p5yr_stochastic_analysis-master/data/'
    par_dir = data_dir + r'par/'
    tim_dir = data_dir + r'tim/'
    noise_file = data_dir + r'channelized_12p5yr_v3_full_noisedict.json' 
    par_files = sorted(glob.glob(par_dir+'*.par'))

    #note par_files has length 46, while tim_files have length 45. Caused by J1713+0747_t2 par_files[21]
    par_files.remove(par_files[21])
    tim_files = sorted(glob.glob(tim_dir+'*.tim'))

    #lists used
    par_name_list = []   #pulsar names generated from the .par files
    tim_name_list = []   #pulsar names generated from the .tim files
    noise_name_list = [] #pulsar names generated from the noise .JSON file
    psr_name_list = []   #offical pulsar list generated from the intersection between the pulsar names found in .tim, .par. and noise .JSON file
    log_10_A__label_list = []  #temporary storage for log_10_A red noise parameter
    gamma_label_list = []      #temporary storage for gamma red noise parameter     
    noise = {}           #raw dictionary of all entires found in the noise .JSON file
    #list of dictionaries of the red noise parameters in form: NAME: (log_10_A, gamma)
    raw_white_noise = []     #future list of white noise values. Could be a branch within a branch to where a specific type can be chosen

    #Uploading noise JSON file and loading it to the noise dictionary
    with open(noise_file,'r') as line:
            noise.update(json.load(line))
    noise_labels = sorted(list(noise.keys()))   #don't use np.unique here cause it is a dictionary key

    #Grabbing pulsar names from .par, .tim, and noise .JSON files
    for par_name in par_files:
        par_name_list.append(get_psrname(par_name))

    for tim_name in tim_files:
        tim_name_list.append(get_psrname(tim_name))

    for noise_label in noise_labels:
        noise_name_list.append(get_psrname(noise_label))

    #This is required to remove all redundent pulsar names, and re-organize it as a sorted list
    noise_name_list = np.unique(noise_name_list)

    #Finds intersection between three lists. I had to do it this way and not the more efficent way from down below due to a duplicate name
    for i in range(len(par_name_list)):
        for j in range(len(tim_name_list)):
            for k in range(len(noise_name_list)):
                if par_name_list[i] == tim_name_list[j] and tim_name_list[j] == noise_name_list[k]:   #The intersection between all three lists
                    psr_name_list.append(par_name_list[i])

    #removes any duplicates using set, rewrites it as a list, and organizes the list
    psr_name_list = np.unique(psr_name_list)
    #This value is a standard number of 45 used repeatedly throughout the code.
    num = len(psr_name_list)

    #removing files so that the data only contains pulsars that they both have info on
    par_files = [f for f in par_files if get_psrname(f) in psr_name_list]
    tim_files = [f for f in tim_files if get_psrname(f) in psr_name_list]
    #try to figure out a way to filer the noise dictionaries here

    for i in range(num):
        for noise_label in noise_labels:  
            #excludes stars found in noise labels, but not in .tim or .par files
            if psr_name_list[i] == get_psrname(noise_label):
                #grabbing red noise parameter keys
                if noise_label == psr_name_list[i] + '_red_noise_log10_A':
                    log_10_A__label_list.append(noise_label)
                    print("red noise parameter log_10_A discovered!\n")
                elif noise_label == psr_name_list[i] + '_red_noise_gamma':
                    gamma_label_list.append(noise_label)
                    print("red noise parameter gamma discovered!\n")
                else:
                    raw_white_noise.append(noise_label)
                    print("white noise parameter discovered\n")
            else:
                print(f"Star {get_psrname(noise_label)} is not found in the .tim or .par files")

    raw_white_noise = np.unique(raw_white_noise)
    
    #organization of noises
    red_noise = []    
    white_noise = []
    for i in range(num):
        red_item= {psr_name_list[i]: (noise[log_10_A__label_list[i]],noise[gamma_label_list[i]])}
        stor = []
        for noises in raw_white_noise:
            if psr_name_list[i] == get_psrname(noises):
                white_item = {noises: noise[noises]}
                stor.append(white_item)
        red_noise.append(red_item)
        white_noise.append({psr_name_list[i]: stor})

    return par_files, tim_files, noise, psr_name_list, red_noise, white_noise



################################################################################################################
################################################################################################################
################################################################################################################
if __name__ == '__main__':
    

    null_time = time.time()
    Ncal_time_file.write(f'{null_time}\n')
    kill_count = 34
    thin = 1
    #code under this is profiled
    
    with cProfile.Profile() as pr:
    #all information tied to the 11-year dataset
        psr_list, pars, tims, noise, rn_psrs = yr_11_data()

        #creates enterprise pulsar
        ePsrs = pulsar_class(pars, tims)
        Tspan = hsen.get_Tspan(ePsrs)
        fyr = 1/(365.25*24*3600)
        freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),200)

        #computes hasasia, and hasasia spectra pulsar to get characteristic strain
        
        #psrs_names_r, h_c_list_r,  freqs_list_r, NcalInv_r = rrf_array_construction(ePsrs)
        psrs_names, h_c_list,  freqs_list, NcalInv = array_construction(ePsrs)
        psrs_names_r, h_c_list_r,  freqs_list_r, NcalInv_r = rrf_array_construction(ePsrs)
        
     
        #for i in range(len(psrs_names)):
        #   for j in range(len(NcalInv[i])):
        #        diff = (NcalInv_r[i]-NcalInv[i])
        #        print(diff)
        



        #print(NcalInv_r[0]-NcalInv[0])

        with open(path + '/test_time.txt', "w") as file:
            stats = pstats.Stats(pr, stream=file)
            stats.sort_stats(pstats.SortKey.CUMULATIVE)
            stats.print_stats()

        for i in range(len(psrs_names_r)):
            plt.loglog(freqs_list[i],h_c_list[i],lw=2,label=psrs_names[i])
            plt.loglog(freqs_list_r[i],h_c_list_r[i],lw=2,label=f'{psrs_names_r[i]} rrf')
        plt.rc('text', usetex=True)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Characteristic Strain, $h_c$')
        plt.legend(loc='upper left')
        plt.show()
        plt.savefig(path+'/h_c_RRF.png')
        plt.close()

        #for i in range(len(psrs_names)):
        #    plt.loglog(freqs_list[i],h_c_list[i],lw=2,label=psrs_names[i])
        #plt.rc('text', usetex=True)
        #plt.xlabel('Frequency [Hz]')
        #plt.ylabel('Characteristic Strain, $h_c$')
        #plt.legend(loc='upper left')
        #plt.show()
        #plt.savefig(path+'/h_c.png')
        #plt.close()
