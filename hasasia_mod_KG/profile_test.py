#Kyle Gourlie
#8/2/2023
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import glob, pickle, json, cProfile, pstats, psutil, sys, os
import matplotlib as mpl
import healpy as hp
import astropy.units as u
import astropy.constants as c
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [5,3]
mpl.rcParams['text.usetex'] = True
from enterprise.pulsar import Pulsar as ePulsar
from memory_profiler import profile, LogFile


sys.path.append('/home/gourliek/hasasia_clone/hasasia')
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky




#memory profile files
path = r'/home/gourliek/Desktop/Profile_Data'
os.mkdir(path)
corr_matrix_mem = open(path + '/corr_matrix_mem_profiler.log','w+')
sens_mem = open(path + '/sens_mem_profiler.log','w+')
h_spectra_mem = open(path + '/h_spectra_profiler.log','w+')





def get_psrname(file,name_sep='_'):
    return file.split('/')[-1].split(name_sep)[0]



def pulsar_class(parameters, timons):
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
    corr_matrix_mem.write(f'Pulsar: {psr.name}\n')
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas,psr.toaerrs,
                                         flags=psr.flags['f'],dt=1)
    keys = [ky for ky in noise.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl,dtype=float)
    for be in backends:
        mask = np.where(psr.flags['f']==be)
        key_ef = '{0}_{1}_{2}'.format(psr.name,be,'efac')
        key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'equad')
        sigma_sqr[mask] = (noise[key_ef]**2 * (psr.toaerrs[mask]**2)
                           + (10**noise[key_eq])**2)
        mask_ec = np.where(fl==be)
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)] 
    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J #ISSUE HERE WHEN RUNNING J1713
    return corr


@profile(stream=sens_mem)
def array_contruction(epsrs):
    psrs = []
    thin = 1
    for ePsr in epsrs:
        #it is dying here
        corr = make_corr(ePsr)[::thin,::thin]
        plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)
        if ePsr.name in rn_psrs.keys():
            Amp, gam = rn_psrs[ePsr.name]
            plaw += hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
        corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                    toas=ePsr.toas[::thin])
        psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                            toaerrs=ePsr.toaerrs[::thin],
                            phi=ePsr.phi,theta=ePsr.theta, 
                            N=corr, designmatrix=ePsr.Mmat[::thin,:])
        psr.name = ePsr.name
        psrs.append(psr)
        del ePsr
    return psrs


@profile(stream=h_spectra_mem)
def hasasia_spectrum(pulsars):
    Specs = []
    for p in pulsars:
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        Specs.append(sp)
        print('\rPSR {0} complete'.format(p.name),end='',flush=True)
    return Specs




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





if __name__ == '__main__':
    kill_count = 3
    #code under this is profiled
    with cProfile.Profile() as pr:
        psr_list, pars, tims, noise, rn_psrs = yr_11_data()
        ePsrs = pulsar_class(pars, tims)
        Tspan = hsen.get_Tspan(ePsrs)
        fyr = 1/(365.25*24*3600)
        freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),300)
        Psrs = array_contruction(ePsrs)
        specs = hasasia_spectrum(Psrs)
        with open(path + '/test_time.txt', "w") as file:
            stats = pstats.Stats(pr, stream=file)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()

    corr_matrix_mem.close()
    sens_mem.close()
    h_spectra_mem.close()
        #for sp,p in zip(specs,Psrs):
            #plt.loglog(sp.freqs,sp.h_c,lw=2,label=p.name)
        

        #plt.legend()
        #plt.show()
        #plt.close()'''

   



  