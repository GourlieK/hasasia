#Kyle Gourlie
#7/23/2024
#library imports
import os, shutil, psutil, time, threading, random, h5py, gc, glob, json, cProfile, pstats, jax
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp
import astropy.units as u
import astropy.constants as c
#mpl.rcParams['figure.dpi'] = 300
#mpl.rcParams['figure.figsize'] = [5,3]
#mpl.rcParams['text.usetex'] = True
from enterprise.pulsar import Pulsar as ePulsar
from memory_profiler import profile

#Memory Profile File Locations
path = os.path.expanduser('~/Desktop/Profile_Data')
#creation of folder to store profile data
try:
    os.mkdir(path)
except FileExistsError:
    pass
    #shutil.rmtree(path)
    #os.mkdir(path)

jax.config.update('jax_platform_name', 'cpu')
from enterprise.pulsar import Pulsar as ePulsar
from memory_profiler import profile

def log_memory_usage(file_path:str):
    """Function to save memory and time profile data at 0.5 second increments, and saves data to txt file

    Args:
        file_path (str): directory of text file
    """
    with open(file_path, 'a') as f:
        while True:
            timestamp = time.time()
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / (1024 ** 3)  # Convert bytes to GB
            f.write(f"{timestamp},{memory_usage}\n")
            f.flush()  
            time.sleep(0.5)

#hasasia imports that are modified so they can be benchmarked
import sensitivity as hsen
import sim as hsim
import skymap as hsky
#needed to profile within sensitivity library
#from sensitivity import get_NcalInv_mem, get_NcalInv_RFF_mem, corr_from_psd_mem

class PseudoPulsar:
    """Quick class to store data from HDF5 file in prep for hasasia pulsar creation"""
    def __init__(self, toas, toaerrs, phi, theta, pdist, N, Mmat=None):
        self.N = N
        self.Mmat = Mmat
        self.phi = phi
        self.theta = theta
        self.toas = toas
        self.toaerrs = toaerrs
        self.pdist = pdist

class PseudoSpectraPulsar:
    """Quick class to store data from HDF5 in prep for hasasia spectrum pulsar creation"""
    def __init__(self, toas, toaerrs, phi, theta, pdist, K_inv, G, designmatrix):
        self.K_inv = K_inv
        self.G = G
        self.phi = phi
        self.theta = theta
        self.toas = toas
        self.toaerrs = toaerrs
        self.pdist = pdist
        self.designmatrix = designmatrix

#white noise covariance matrix profile file
corr_matrix_mem = open(path + '/corr_matrix_mem.txt','a')
#creation of hasasia pulsar
hsen_psr_mem = open(path + '/psr_hsen_mem.txt','a')
#Rank Reduced Formalism creation of hasasia pulsar
hsen_psr_RRF_mem= open(path + '/psr_hsen_mem_RRF.txt','a')
#
hsen_psr_spec_mem = open(path + '/psr_hsen_spec_mem.txt','a')
#Rank Reduced Formalism creation of hasasia pulsar
hsen_psr_RRF_spec_mem= open(path + '/psr_hsen_spec_mem_RRF.txt','a')
#Total time taken to generate each pulsar 
time_inc_psr = open(path + '/psr_increm.txt','a')
#Total time taken to generate each spectrum
time_inc_specs = open(path + '/specs_increm.txt','a')

batch_time = open(path + '/Batch_time.txt','a')

def get_psrname(file,name_sep='_'):
    """Function that grabs names of pulsars from parameter files
    
    Returns:
        Pulsar name
    """
    return file.split('/')[-1].split(name_sep)[0]

@profile(stream = corr_matrix_mem)
def make_corr(psr: ePulsar, noise:dict, yr:float)->np.array:
    """_summary_: Computes white noise correlation matrix for a given enterprise.pulsar object

    Args:
        - psr (ePulsar): enterprise.pulsar object
        - noise (dict): white noise parameters with front and backends
        - yr (float): if yr=15, then change key_eq and sigma_sqt

    Returns:
        np.array: white noise correlation matrix
    """
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
        if yr == 15:
            key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'t2equad')
            sigma_sqr[mask] = (noise[key_ef]**2 * ((psr.toaerrs[mask]**2) ## t2equad -- new/correct for 15yr
                           + (10**noise[key_eq])**2))
        else:
            key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'equad')
            sigma_sqr[mask] = (noise[key_ef]**2 * (psr.toaerrs[mask]**2) ## tnequad -- old/wrong but used in 15yr
                            + (10**noise[key_eq])**2)
        mask_ec = np.where(fl==be)
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)]

    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J
    return corr

def enterprise_creation(pars:list, tims:list, ephem:str)->list:
    """_summary_: Generate list of enterprise.pulsars objects

    Args:
        pars (list): list of parameter files 
        tims (list): list of timing files
        ephem (str): ephemeris

    Returns:
        list: list of enterprise.pulsar objects
    """
    enterprise_Psrs = []
    count = 1
    for par,tim in zip(pars,tims):
        if count <= kill_count:
            ePsr = ePulsar(par, tim,  ephem=ephem)
            enterprise_Psrs.append(ePsr)
            print('\rPSR {0} complete'.format(ePsr.name),end='',flush=True)
            print(f'\n{count} pulsars created')
            count +=1
        else:
            break
    return enterprise_Psrs

def enterprise_hdf5(ePsrs:list, noise:dict, yr:float, edir:str, thin):
    """Writes enterprise.pulsar objects onto HDF5 file with WN Covariance matrix attributes.

    - ePsrs (list): List of enterprise.pulsar objects
    - edir: Directory in which to store HDF5 file under
    """
    Tspan = hsen.get_Tspan(ePsrs)
    with h5py.File(edir, 'w') as f:
        Tspan_h5 = f.create_dataset('Tspan', (1,), float)
        Tspan_h5[:] = Tspan
        #numpy array stored with placeholders so names can be indexed into it later, also storing strings as bytes
        name_list = np.array(['X' for _ in range(kill_count)], dtype=h5py.string_dtype(encoding='utf-8'))
        #pseudo while/for-loop designed to delete first entry
        i = 0
        while True:
            ePsrs[0].N = make_corr(ePsrs[0], noise, yr)[::thin, ::thin]
            hdf5_psr = f.create_group(ePsrs[0].name)
            hdf5_psr.create_dataset('toas', ePsrs[0].toas.shape, ePsrs[0].toas.dtype, data=ePsrs[0].toas)
            hdf5_psr.create_dataset('toaerrs', ePsrs[0].toaerrs.shape,ePsrs[0].toaerrs.dtype, data=ePsrs[0].toaerrs)
            hdf5_psr.create_dataset('phi', (1,), float, data=ePsrs[0].phi)
            hdf5_psr.create_dataset('theta', (1,), float, data=ePsrs[0].theta)
            hdf5_psr.create_dataset('designmatrix', ePsrs[0].Mmat.shape, ePsrs[0].Mmat.dtype, data=ePsrs[0].Mmat)
            hdf5_psr.create_dataset('N', ePsrs[0].N.shape, ePsrs[0].N.dtype, data=ePsrs[0].N)
            hdf5_psr.create_dataset('pdist', (2,), float, data=ePsrs[0].pdist)
            name_list[i] = ePsrs[0].name
            f.flush()
            del ePsrs[0]
            i+=1
            #once all the pulsars are deleted, the length of the list is zero
            if len(ePsrs) == 0:
                break
            
        f.create_dataset('names',data = name_list)
        f.flush()
        print('enterprise.pulsars successfully saved to HDF5\n')


def hsen_pulsar_entry(psr:hsen.Pulsar, dir:str):
    """Writes hasasia pulsar object to hdf5 file"""        
    with h5py.File(dir, 'a') as f:
        hdf5_psr = f.create_group(psr.name)
        hdf5_psr.create_dataset('toas', psr.toas.shape, psr.toas.dtype, data=psr.toas)
        hdf5_psr.create_dataset('toaerrs', psr.toaerrs.shape,psr.toaerrs.dtype, data=psr.toaerrs)
        hdf5_psr.create_dataset('phi', (1,), float, data=psr.phi)
        hdf5_psr.create_dataset('theta', (1,), float, data=psr.theta)
        hdf5_psr.create_dataset('designmatrix', psr.designmatrix.shape, psr.designmatrix.dtype, data=psr.designmatrix)
        hdf5_psr.create_dataset('G', psr.G.shape, psr.G.dtype, data=psr.G)
        hdf5_psr.create_dataset('K_inv', psr.K_inv.shape, psr.K_inv.dtype, data=psr.K_inv)
        hdf5_psr.create_dataset('pdist', (2,), float, data=psr.pdist)
        f.flush()
        print(f'hasasia pulsar {psr.name} successfully saved to HDF5', end='\r')

@profile(stream=hsen_psr_RRF_mem)
def hsen_pulsar_rrf_creation(pseudo: PseudoPulsar, hsen_dir_rrf:str):
    """_summary_: create hasasia pulsar object using rank-reduced method

    Args:
        pseudo (PseudoPulsar): PseudoPulsar object
        hsen_dir_rrf (str): directory for storing hasasia pulsar object
    """
    start_time = time.time()
    psr = hsen.Pulsar(toas=pseudo.toas[::thin],
                                    toaerrs=pseudo.toaerrs[::thin],
                                    phi=pseudo.phi,theta=pseudo.theta, 
                                    N=pseudo.N, designmatrix=pseudo.Mmat[::thin,:], pdist=pseudo.pdist)
    psr.name = pseudo.name
    _ = psr.K_inv
    end_time = time.time()
    time_inc_psr.write(f"RRF {psr.name} {start_time-null_time} {end_time-null_time}\n")
    time_inc_psr.flush()
    hsen_pulsar_entry(psr, hsen_dir_rrf)

def hsen_rrf_pulsar_hdf5_entire(f:str, names_list:list, hsen_dir_rrf:str):
    """_summary_: Function that goes through entire process of creating fake hasasia pulsar object, create hasasia pulsar object, and 
    saves attributes to hdf5 file. Primary use of this function is to be not executed if already saved. This function is for the 
    rank-reduced method

    Args:
        f (str): enterprise pulsar hdf5 directory
        names_list (list): list of pulsar names
        hsen_dir_rrf (str): pulsar hdf5 directory that will used to save the required attributes
    """
    for name in names_list:
        psr = f[name]
        pseudo = PseudoPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                        theta = psr['theta'][:][0], pdist=psr['pdist'][:], N=psr['N'][:])
        pseudo.name = name
        pseudo.Mmat= psr['designmatrix'][:]

        hsen_psr_RRF_mem.write(f'Pulsar: {name}\n')
        hsen_psr_RRF_mem.flush()
        hsen_pulsar_rrf_creation(pseudo, hsen_dir_rrf)
        del pseudo

@profile(stream=hsen_psr_spec_mem)
def hsen_spectrum_creation(pseudo:PseudoSpectraPulsar)->hsen.Spectrum:
    """_summary_: Creates Spectrum object using the original method

    Args:
        pseudo (PseudoSpectraPulsar): fake spectrum pulsar that contains all needed attributes

    Returns:
        hsen.Spectrum: spectrum object
    """
    start_time = time.time()
    spec_psr = hsen.Spectrum(pseudo, freqs=freqs)
    spec_psr.name = pseudo.name
    #Calling computation of NcalInv, due to its high computational cost
    _ = spec_psr.NcalInv
    end_time = time.time()
    time_inc_specs.write(f"OG {spec_psr.name} {start_time-null_time} {end_time-null_time}\n")
    time_inc_specs.flush()
    return spec_psr

@profile(stream=hsen_psr_RRF_spec_mem)
def hsen_spectrum_creation_rrf(pseudo:PseudoSpectraPulsar, gam_gw, A_gw, A_irn, gam_irn)-> hsen.RRF_Spectrum:
    """_summary_: Creates Spectrum object using the rank-reduced method

    Args:
        pseudo (PseudoSpectraPulsar): fake spectrum pulsar that contains all needed attributes

    Returns:
        hsen.RRF_Spectrum: spectrum object
    """
    start_time = time.time()
    spec_psr = hsen.RRF_Spectrum(psr=pseudo, freqs_gw=freqs_gwb,amp_gw=A_gw, gamma_gw=gam_gw,
                                     freqs_rn=freqs_rn, amp = A_irn, gamma = gam_irn, freqs=freqs)     
    spec_psr.name = pseudo.name

    _ = spec_psr.NcalInv
    end_time = time.time()
    time_inc_specs.write(f"RRF {spec_psr.name} {start_time-null_time} {end_time-null_time}\n")
    time_inc_specs.flush()
    return spec_psr


def yr_11_data():
    """Creates enterprise pulsars from the 11 yr dataset from parameter and timing files.

    The quantities that are being returned within this function will be attributes used to write 
    enterprise pulsars onto HDF5 file for 

    Returns:
        - psr_list (list): List of pulsars names
        - enterprise_Psrs (list): List of enterprise pulsars 
        - noise (dict): Noise parameters including fe/be of WN and RN.
        index is spectral amplitude and 1st index is spectral index
        - Tspan: Total timespan of the PTA
        - enterprise_dir: specific directory name used for enterprise HDF5 file
    """
    
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

    edir = '/home/gourliek/11_yr_enterprise_pulsars.hdf5'
    ephem = 'DE436'
    return pars, tims, noise, edir, ephem

def yr_12_data():
    """Creates enterprise pulsars from the 12.5 yr dataset from parameter and timing files.

    The quantities that are being returned within this function will be attributes used to write 
    enterprise pulsars onto HDF5 file for 

    Returns:
        - psr_list (list): List of pulsars names
        - enterprise_Psrs (list): List of enterprise pulsars 
        - noise (dict): Noise parameters including fe/be of WN and RN.
        index is spectral amplitude and 1st index is spectral index
        - Tspan: Total timespan of the PTA
        - enterprise_dir: specific directory name used for enterprise HDF5 file
    """

    data_dir = r'/home/gourliek/Nanograv/12p5yr_stochastic_analysis-master/data/'
    par_dir = data_dir + r'par/'
    tim_dir = data_dir + r'tim/'
    noise_file = data_dir + r'channelized_12p5yr_v3_full_noisedict.json' 

    #directory name of enterprise hdf5 file
    
    
    
    #sorting parameter and timing files
    parfiles = sorted(glob.glob(par_dir+'*.par'))
    timfiles = sorted(glob.glob(tim_dir+'*.tim'))

    #getting names of pulsars from timing files
    par_psr_names = []
    for file in parfiles:
        par_psr_names.append(get_psrname(file))

    #getting names of pulsars from parameter files
    tim_psr_names = []
    for file in timfiles:
        tim_psr_names.append(get_psrname(file))

    #grabbing intersection of names
    psr_list= [item for item in tim_psr_names if item in par_psr_names]
    
    pars_v1 = [f for f in parfiles if get_psrname(f) in psr_list]

     # ...filtering out the tempo parfile...
    pars = [x for x in pars_v1 if 'J1713+0747_NANOGrav_12yv3.gls.par' not in x]
    tims = [f for f in timfiles if get_psrname(f) in psr_list]

    noise = {}
    with open(noise_file, 'r') as fp:
        noise.update(json.load(fp))

    edir = '/home/gourliek/12_yr_enterprise_pulsars.hdf5'
    ephem = 'DE438'
    
    return pars, tims, noise, edir, ephem

def chains_puller(num: int, names_list:list):
    """_summary_: Reads generated chains from the 12.5 yr data, varying spectral index, 30 frequencies, from the DE438 ephemeris.

    Resource: https://nanograv.org/science/data/125-year-stochastic-gravitational-wave-background-search

    Args:
        num (int): number of random samples
        names_list (list): List of pulsar names

    Returns:
        Two dictionaries containing instrinsic red noise parameters
    """
    chain_path = os.path.expanduser('~/Nanograv/12p5yr_varying_sp_ind_30freqs/12p5yr_DE438_model2a_cRN30freq_gammaVary_chain.hdf5')
    with h5py.File(chain_path, 'r') as chainf:
        
        params = chainf['params'][:]
        samples = np.array(chainf['samples'][:])
        
        
    list_params = [item.decode('utf-8') for item in params]
    list_params.remove('gw_gamma')
    list_params.remove('gw_log10_A')
    irn_params_log10_A = {}
    irn_params_gam = {}

    for name in names_list:
        log10_A_ind = list_params.index(name+'_red_noise_log10_A')
        gamma_ind = list_params.index(name+'_red_noise_gamma')
        log10_A_samples = samples[30000:,log10_A_ind]
        gamma_samples = samples[30000:,gamma_ind]
        rand_samples_ind = np.random.choice(a=log10_A_samples.shape[0], size=num, replace=False)
        irn_params_log10_A[name] = log10_A_samples[rand_samples_ind]
        irn_params_gam[name] = gamma_samples[rand_samples_ind]

    return irn_params_log10_A, irn_params_gam


def save_h_c(data_path, hc_dsc, hc_sc, a_gw, gam_gw, freqs, batch_num: int): 
    with h5py.File(data_path, 'a') as f:
        hdf5 = f.create_group(f'batch_{batch_num}')
        hdf5.create_dataset('A_irn', (1,), float, data=a_gw)
        hdf5.create_dataset('gam_irn', (1,), float, data=gam_gw)
        hdf5.create_dataset('hc_dsc', hc_dsc.shape, hc_dsc.dtype, data=hc_dsc)
        hdf5.create_dataset('hc_sc', hc_sc.shape, hc_sc.dtype, data=hc_sc)
        hdf5.create_dataset('freqs', freqs.shape, freqs.dtype, data=freqs)
        f.flush()

def sens_gen():
    log10_A_samples, gam_samples = chains_puller(num_chains, names_list)
    for i in range(num_chains):
        time_start_bt = time.time()
        
        #reading hdf5 file containing hasasia pulsar attributes from rank-reduced method to create list of spectrum objects     
        with h5py.File(hsen_dir_rrf,'r') as hsenfrrf:
            specs_rrf = []
            for name in names_list:
                psr = hsenfrrf[name]
                pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                        theta = psr['theta'][:][0], pdist=psr['pdist'][:], K_inv=psr['K_inv'][:], G=psr['G'][:],
                                        designmatrix=psr['designmatrix'])
                pseudo.name = name
                psr_log10_A = log10_A_samples[pseudo.name][i]
                psr_gam = gam_samples[pseudo.name][i]
                print(f'IRN Spectral Index:{psr_gam}')
                print(f'IRN Spectral Amplitude:{10**psr_log10_A}')
                hsen_psr_RRF_spec_mem.write(f'Pulsar: {name}\n')
                hsen_psr_RRF_spec_mem.flush()
                spec_psr_rrf = hsen_spectrum_creation_rrf(pseudo, gam_gw=gam_gw, A_gw= A_gw, A_irn=10**psr_log10_A, gam_irn=psr_gam)
                specs_rrf.append(spec_psr_rrf)
                
        #creation of sensitivity curves rank-reduced method
        rrf_sc = hsen.GWBSensitivityCurve(specs_rrf)
        rrf_dsc = hsen.DeterSensitivityCurve(specs_rrf)
        del specs_rrf
        save_h_c(data_path, rrf_dsc.h_c, rrf_sc.h_c,
                    10**psr_log10_A, psr_gam, rrf_sc.freqs, i)
        del rrf_sc
        del rrf_dsc
        print()
        print(f'Batch #{i+1} finished!')
        print()
        time_end_bt = time.time()
        batch_time.write(f'{time_end_bt-time_start_bt}\n')
        batch_time.flush()
    batch_time.close()


def sigma_grab(sigma:int):
    dz = 0.01
    input = np.arange(-sigma, sigma+dz, dz)
    output = np.exp(-0.5*input**2)
    sigma_val = np.trapz(y=output, x=input, dx=dz)/np.sqrt(2*np.pi)
    lower = .5 - sigma_val/2
    upper = .5 + sigma_val/2
    return lower, upper




if __name__ == '__main__':
    null_time = time.time()
    log_path = path+'/time_mem_data.txt'
    #this will allow memory profile data to run in parallel with the rest of the program
    logging_thread = threading.Thread(target=log_memory_usage, args=(log_path,))
    logging_thread.daemon = True  # Ensure the thread will exit when the main program exits
    logging_thread.start()

    ###################################################
    #max is 34 for 11yr dataset
    #max is 45 for 12yr dataset
    kill_count = 45
    num_chains = 50
    thin = 20
    A_gw = 1.73e-15
    gam_gw = 13/3

    #yr used for making WN correlation matrix, specifically when yr=15
    yr=12
    fyr = 1/(365.25*24*3600)

    names_list = []
    with cProfile.Profile() as pr:
        #Realistic PTA datasets
        #pars, tims, noise, edir, ephem = yr_11_data()
        pars, tims, noise, edir, ephem = yr_12_data()

        #if file does not exist, then re-compute it
        if not os.path.isfile(edir):
            ePsrs = enterprise_creation(pars, tims, ephem)
            enterprise_hdf5(ePsrs, noise, yr, edir, thin)
            del ePsrs

        #reading hdf5 file containing enterprise.pulsar attributes
        with h5py.File(edir, 'r') as f:
            #reading Tspan and creation of frequencies to observe
            Tspan = f['Tspan'][:][0]

            freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),400)
            freqs_rn = np.linspace(1/Tspan, 30/Tspan, 30)
            freqs_gwb = np.linspace(1/Tspan, 14/Tspan, 14)

            #reading names encoded as bytes, and re-converting them to strings, and deleting byte names
            names = f['names'][:]
            for i in range(kill_count):
                names_list.append(names[i].decode('utf-8'))
            del names    
                
            #Rank-Reduced Method for creation of hasasia pulsars, and saving them to hdf5 file
            hsen_dir_rrf = os.path.expanduser('~/hsen_psrs_rrf.hdf5')
            if not os.path.isfile(hsen_dir_rrf):
                hsen_rrf_pulsar_hdf5_entire(f, names_list, hsen_dir_rrf)

        data_path = os.path.expanduser(path+'/h_c_data.hdf5')
        if not os.path.isfile(data_path):
            sens_gen()
        

    with open(path + '/test_time.txt', 'w') as file:
        #saving cprofile data
        stats = pstats.Stats(pr, stream=file)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

        h_sc_matrix = np.zeros((freqs.size, num_chains))
        h_dsc_matrix = np.zeros((freqs.size, num_chains))
    
        
        #plotting sensitivity curves
        
        for i in range(num_chains):
            with h5py.File(data_path, 'r') as data_f:
                data = data_f['batch_'+str(i)]
                freqs = data['freqs'][:]
                h_sc_matrix[:,i] = data['hc_sc'][:]
                h_dsc_matrix[:,i] = data['hc_dsc'][:]

    h_sc_sigma_low = []
    h_sc_med = []
    h_sc_sigma_high = []

    h_dsc_sigma_low = []
    h_dsc_med = []
    h_dsc_sigma_high = []

    low, high = sigma_grab(sigma=1)
    
    
    for i in range(freqs.size):
        h_sc_sigma_low.append(np.quantile(a=h_sc_matrix[i,:], q=low))
        h_sc_med.append(np.median(h_sc_matrix[i,:]))
        h_sc_sigma_high.append(np.quantile(a=h_sc_matrix[i,:], q=high))

        h_dsc_sigma_low.append(np.quantile(a=h_dsc_matrix[i,:], q=low))
        h_dsc_med.append(np.median(h_dsc_matrix[i,:]))
        h_dsc_sigma_high.append(np.quantile(a=h_dsc_matrix[i,:], q=high))


    plt.title(f'NANOGrav {yr}-year Data Set Sensitivity Curve')
    plt.loglog(freqs, h_sc_med, label='Median Stoch', lw=0.2, c='black')
    plt.loglog(freqs, h_dsc_med, label='Median Det', lw=0.2, c='red')
    

    plt.fill_between(freqs, h_sc_sigma_low, h_sc_sigma_high, color='gray', label='1$\sigma$ Stoch')
    plt.fill_between(freqs, h_dsc_sigma_low, h_dsc_sigma_high, color='orange', label='1$\sigma$ Det')

    
    plt.tight_layout()
    plt.xlabel('Frequencies, Hz')
    plt.ylabel('Characteristic Strain, $h_c$')
    plt.grid(which='both')
    
    plt.axvline(x=1/Tspan, label=r'$\frac{1}{\mathrm{Tspan}}$', c='blue', linestyle='--')
    plt.axvline(x=14/Tspan, c='cyan', label=r'$\frac{14}{\mathrm{Tspan}}$', linestyle='--')
    plt.axvline(x=30/Tspan, label=r'$\frac{30}{\mathrm{Tspan}}$', c='green', linestyle='--')
    plt.legend()
    plt.savefig(path+'/sc_h_c_sigma.png', dpi=1000)
    plt.show()
    plt.close()

    plt.title(f'NANOGrav {yr}-year Data Set Sensitivity Curve')
    for i in range(num_chains):
        plt.loglog(freqs, h_sc_matrix[:,i], label='Stoch', c='black', lw=0.2)
        plt.loglog(freqs, h_dsc_matrix[:,i], label='Det', c='black', lw=0.2)

    plt.tight_layout()
    plt.xlabel('Frequencies, Hz')
    plt.ylabel('Characteristic Strain, $h_c$')
    plt.grid(which='both')
    
    plt.axvline(x=1/Tspan, label=r'$\frac{1}{\mathrm{Tspan}}$', c='blue', linestyle='--')
    plt.axvline(x=14/Tspan, c='cyan', label=r'$\frac{14}{\mathrm{Tspan}}$', linestyle='--')
    plt.axvline(x=30/Tspan, label=r'$\frac{30}{\mathrm{Tspan}}$', c='green', linestyle='--')
    #plt.legend()
    plt.savefig(path+'/sc_h_c_total.png', dpi=1000)
    plt.show()
    plt.close()

    
    
    bt_data = []
    with open(path + '/Batch_time.txt', 'r') as file:
        for line in file:
            bt_data.append(float(line))

    batch_num = [i+1 for i in range(len(bt_data))]
    plt.figure(dpi=100)  # Set the DPI to 100 for better resolution
    plt.bar(batch_num, bt_data, color='skyblue')

    # Add labels and title
    plt.xlabel('Batch Number')
    plt.ylabel('Batch Time [s]')
    plt.title('Batches vs Time')

    #Show the bar chart
    plt.tight_layout()
    plt.savefig(path+'/Batch_time_psrs.png', dpi=1000)
    plt.show()
    plt.close()


##############################MEMORY VS TIME PLOTTING START#################################################
    #total memory vs time data
    mem_data = []
    time_data = []
    with open(log_path, 'r') as file:  
        for line in file:
            line_sep = line.split(',')
            time_data.append(float(line_sep[0]))
            mem_data.append(float(line_sep[1]))

    time_data = np.array(time_data)
    time_data = time_data-null_time
    mem_data = np.array(mem_data)
    plt.style.use('dark_background')
    plt.title('Memory vs Time')
    plt.plot(time_data, mem_data, c='yellow')
    plt.xlabel('Time (s)')
    plt.ylabel('Virtual Memory (GB)')
    plt.savefig(path+'/mem_time.png', dpi=1000)
    plt.show()
    plt.close()
##############################MEMORY VS TIME PLOTTING END#######################################################


   