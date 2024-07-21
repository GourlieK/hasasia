#Kyle Gourlie
#7/21/2024
#library imports
import shutil, h5py
import gc
import numpy as np
import dask.array as da
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



#Memory Profile File Locations
path = os.path.expanduser('~/Profile_Data')
try:
    os.mkdir(path)
except FileExistsError:
    shutil.rmtree(path)
    os.mkdir(path)
#white noise covariance matrix profile file
corr_matrix_mem = open(path + '/corr_matrix_mem.txt','w')
#creation of hasasia pulsar, covariance, then writing to disk
hsen_psr_mem = open(path + '/psr_hsen_mem.txt','w')
#Rank Reduced Formalism creation of hasasia pulsar, covaraince, then writing to disk
hsen_psr_RRF_mem= open(path + '/psr_hsen_mem_RRF.txt','w')
#Original method for creation of spectra pulsar
hsen_psr_spec_mem = open(path + '/psr_hsen_spec_mem.txt','w')
#Rank Reduced Formalism creation of spectra pulsar
hsen_psr_RRF_spec_mem= open(path + '/psr_hsen_spec_mem_RRF.txt','w')


#Total time taken to generate each pulsar 
time_increments = open(path + '/psr_increm.txt','w')
#Default profile file to generate NcalInv (bottle neck for computation)
NcalInv_time_file = open(path + '/NcalInv_time.txt','w')
#Rank Reduced Formalism file to generate NcalInv (bottle neck for computation)
NcalInv_RRF_time_file = open(path + '/NcalInv_time_RRF.txt','w')
Null_time_file = open(path + '/Null_time.txt','w')



#hasasia imports that are modified so they can be benchmarked
import sensitivity as hsen
import sim as hsim
import skymap as hsky
#needed to profile within sensitivity library
from sensitivity import get_NcalInv_mem, get_NcalInv_RFF_mem, corr_from_psd_mem



class PseudoPulsar:
    """Quick class to store data from HDF5 file in prep for hasasia pulsar creation"""
    def __init__(self, name, toas, toaerrs, phi, theta, pdist, N, designmatrix=None):
        self.name = name
        self.designmatrix = designmatrix
        self.phi = phi
        self.theta = theta
        self.toas = toas
        self.toaerrs = toaerrs
        self.pdist = pdist
        self.N = N

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



def get_psrname(file,name_sep='_'):
    """Function that grabs names of pulsars from parameter files
    
    Returns:
        Pulsar name
    """
    return file.split('/')[-1].split(name_sep)[0]






@profile(stream = corr_matrix_mem)
def make_corr(psr: ePulsar, noise:dict)->np.array:
    """Calculates the white noise covariance matrix based on EFAC, EQUAD, and ECORR

    Args:
        psr (object): Enterprise Pulsar Object

    Returns:
        corr (array): white noise covariance matrix
    """
    corr_matrix_mem.write(f"{psr.name}\n")
    corr_matrix_mem.flush()
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
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)] 
    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J
    return corr

@profile(stream=hsen_psr_mem)
def hsen_creation_to_hdf5(ePsr):
    """Function that takes enterprise fake pulsar as input, computes noise covariance matrix, creates hasasia.pular object, and
    sends to function to write to hdf5 using original method"""
    hsen_psr_mem.write(f"{ePsr.name}\n")
    hsen_psr_mem.flush()
    #white noise corvariance matrix
    #adding power spectral density form GWB
    plaw = hsen.red_noise_powerlaw(A=A_gw, gamma=gamma_gw, freqs=freqs_gw)

    if ePsr.name in rn_psrs.keys():
        Amp, gam = rn_psrs[ePsr.name]
        #adding power spectral density from pulsar intrinsic noise
        plaw += hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs) 

    #adding red noise components to the noise covariance matrix via power spectral density
    ePsr.N += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                toas=ePsr.toas[::thin])   

    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta, 
                        N=ePsr.N, designmatrix=ePsr.designmatrix[::thin,:], pdist=ePsr.pdist)
    #need to add name of pulsar
    psr.name = ePsr.name
    #quantities that are inherit to hsen.pulsar
    _ = psr.K_inv
    __ = psr.G

    directory = os.path.expanduser('~/hsen_psrs_OG.hdf5')
    #comment this out if hasasia pulsar hdf5 is already computed
    ############################################################
    new_dir = hsen.hsen_psr_to_hdf5(psr, directory)
    del directory
    ############################################################
    print(f"Hasasia Pulsar {psr.name} created\n")
    del psr
    return new_dir
   


@profile(stream=hsen_psr_RRF_mem)
def hsen_creation_to_hdf5_rrf(ePsr):
    """Function that takes enterprise fake pulsar as input, computes noise covariance matrix, creates hasasia.pular object, and
    sends to function to write to hdf5 using rank reduced formalism"""
    hsen_psr_RRF_mem.write(f"{ePsr.name}\n")
    hsen_psr_RRF_mem.flush()
    #white noise corvariance matrix
    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta, 
                        N=ePsr.N, designmatrix=ePsr.designmatrix[::thin,:], pdist=ePsr.pdist)
    
    psr.name = ePsr.name
    #quantities that are inherit to hsen.pulsar
    _ = psr.K_inv
    __ = psr.G

    directory = os.path.expanduser('~/hsen_psrs_RRF.hdf5')
    ##############################################################
    #comment this out if hasasia pulsar hdf5 is already computed
    new_dir = hsen.hsen_psr_to_hdf5(psr, directory)
    del directory
    #############################################################
    print(f"Hasasia RRF Pulsar {psr.name} created\n")
    del psr
    return new_dir

   
    

@profile(stream=hsen_psr_spec_mem)
def hsen_spectra_creation(freqs, names, path)->list:
    """Creation of hasasia spectrum pulsars 

    Args:
        - freqs (numpy.array): frequencies
        - names (list): list containing all names of pulsars

    Returns:
        - spectras (list): list of Hasasia Spectrum Pulsars
    """
    
    spectras = []
    with h5py.File(path, 'r') as f:
        for name in names:
            start_time = time.time()
            psr = f[name]
            pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                    theta = psr['theta'][:][0], K_inv=psr['K_inv'][:], G=psr['G'][:], pdist=psr['pdist'][:],
                                    designmatrix=psr['designmatrix'])
            pseudo.name = name

            spec_psr = hsen.Spectrum(pseudo, freqs=freqs)
            spec_psr.name = pseudo.name
            print(f'Hasasia Spectrum {spec_psr.name} created\n')

            del pseudo
            ########################################
            #quantity is called up so it is computed
            get_NcalInv_mem.write(f'Pulsar: {name}\n')
            get_NcalInv_mem.flush()
            NcalInv_time_start = time.time()
            _ = spec_psr.NcalInv
            NcalInv_time_end = time.time()
            ########################################
            end_time = time.time()
            NcalInv_time_file.write(f"{name}\t{NcalInv_time_end-NcalInv_time_start}\n")
            time_increments.write(f"OG {name} {start_time-null_time} {end_time-null_time}\n")
            NcalInv_time_file.flush()
            time_increments.flush()
            spectras.append(spec_psr)
    return spectras

    
@profile(stream=hsen_psr_RRF_spec_mem)
def hsen_spectra_creation_rrf(freqs, freqs_gw, names, path)->list:
    """Creation of hasasia spectrum pulsars using RRF

    Args:
        - freqs (numpy.array): frequencies
        - names (list): list containing all names of pulsars

    Returns:
        - spectras (list): list of Hasasia Spectrum Pulsars
    """
    spectras = []
    with h5py.File(path, 'r') as f:
        for name in names:
            start_time = time.time()
            psr = f[name]
            pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                    theta = psr['theta'][:][0], pdist=psr['pdist'][:], K_inv=psr['K_inv'][:], G=psr['G'][:],
                                    designmatrix=psr['designmatrix'])
            pseudo.name = name
            
            if pseudo.name in rn_psrs.keys():
                Amp, gam = rn_psrs[pseudo.name]
                #creates spectrum hasasia pulsar to calculate characteristic straing
                spec_psr = hsen.RRF_Spectrum(pseudo, A_gw=A_gw, gamma_gw=gamma_gw, freqs_gw=freqs_gw,amp = Amp, gamma = gam, freqs=freqs)
                spec_psr.name = pseudo.name
            
            else:
                #creates spectrum hasasia pulsar to calculate characteristic straing
                spec_psr = hsen.RRF_Spectrum(pseudo, A_gw=A_gw, gamma_gw=gamma_gw, freqs=freqs, freqs_gw=freqs)
                spec_psr.name = pseudo.name

            print(f'Hasasia Spectrum RRF {spec_psr.name} created\n')
            #hasasia pulsar no longer needed
            del pseudo
            ########################################
            #quantity is called up so it is computed
            get_NcalInv_RFF_mem.write(f'Pulsar: {name}\n')
            get_NcalInv_RFF_mem.flush()
            NcalInv_time_start = time.time()
            ####
            _ = spec_psr.NcalInv#.compute()
            ####
            NcalInv_time_end = time.time()
            end_time = time.time()
            ########################################
            NcalInv_RRF_time_file.write(f"RRF{name}\t{NcalInv_time_end-NcalInv_time_start}\n")
            time_increments.write(f"RRF {name} {start_time-null_time} {end_time-null_time}\n")
            NcalInv_RRF_time_file.flush()
            time_increments.flush()
            spectras.append(spec_psr)
    return spectras


def yr_11_data():
    """Creates enterprise pulsars from the 11 yr dataset from parameter and timing files.

    The quantities that are being returned within this function will be attributes used to write 
    enterprise pulsars onto HDF5 file for 

    Returns:
        - psr_list (list): List of pulsars names
        - enterprise_Psrs (list): List of enterprise pulsars 
        - noise (dict): Noise parameters including fe/be of WN and RN.
        - rn_psrs (dict): RN parameters where key is name of pulsar and value is list where 0th 
        index is spectral amplitude and 1st index is spectral index
        - Tspan: Total timespan of the PTA
        - enterprise_dir: specific directory name used for enterprise HDF5 file
    """

    #File Paths
    pardir = os.path.expanduser('~/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/')
    timdir = os.path.expanduser('~/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/')
    noise_dir = os.path.expanduser('~/Nanograv/11yr_stochastic_analysis-master')
    noise_dir += '/nano11y_data/noisefiles/'
    psr_list_dir = os.path.expanduser('~/Nanograv/11yr_stochastic_analysis-master/psrlist.txt')

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

    edir = os.path.expanduser('~/11_yr_enterprise_pulsars.hdf5')
    ephem = 'DE436'
    
    
      
    return pars, tims, noise, rn_psrs, edir, ephem



def yr_12_data():
    """Creates enterprise pulsars from the 12.5 yr dataset from parameter and timing files.

    The quantities that are being returned within this function will be attributes used to write 
    enterprise pulsars onto HDF5 file for 

    Returns:
        - psr_list (list): List of pulsars names
        - enterprise_Psrs (list): List of enterprise pulsars 
        - noise (dict): Noise parameters including fe/be of WN and RN.
        - rn_psrs (dict): RN parameters where key is name of pulsar and value is list where 0th 
        index is spectral amplitude and 1st index is spectral index
        - Tspan: Total timespan of the PTA
        - edir: specific directory name used for enterprise HDF5 file
    """

    data_dir = os.path.expanduser('~/Nanograv/12p5yr_stochastic_analysis-master/data/')
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

    #initialize dictionary list with placeholders where parameters for rn will be held
    rn_psrs = {}
    for name in psr_list:
        amp_key = name + '_red_noise_log10_A'
        gamma_key = name + '_red_noise_gamma'
        for key in noise:
            if key == amp_key or key == gamma_key:
                rn_psrs[name] = ['x','x']
    
    #place proper entries
    for name in psr_list:
        amp_key = name + '_red_noise_log10_A'
        gamma_key = name + '_red_noise_gamma'
        for key in noise:
            if key == amp_key:
                rn_psrs[name][0] = 10**noise[amp_key]  #because parameter is log_10()
            elif key == gamma_key:
                rn_psrs[name][1] = noise[gamma_key]

    edir = os.path.expanduser('~/12_yr_enterprise_pulsars.hdf5')
    ephem = 'DE438'
    
    return pars, tims, noise, rn_psrs, edir, ephem


def ent_psr_to_hdf5(ePsrs:list, noise:dict, edir:str=os.path.expanduser('~/enterprise_psrs.hdf5')) ->str:
    """## Function that writes list of enterprise.pulsars objects to disk using hdf5 file format. ##

    The data stored within the hdf5 file are the required attributes needed from each enterprise.pulsar object to create a hasasia.pulsar object,
    along with other important data. The data is organized within the hdf5 file as follows where the type is organized by python type, then hdf5 type:
    - Tspan (float, dataset): total timespan of the PTA dataset
    - enterprise.pulsar.name (str, group): name of pulsar, organized as a group within hdf5 file
    - enterprise.pulsar.toas (numpy.array, dataset of group): array of observatory TOAs in seconds
    - enterprise.pulsar.toaerrs (numpy.array, dataset of group): array of TOA errors in seconds
    - enterprise.pulsar.phi (float, dataset of group): azimuthal angle of pulsar in radians
    - enterprise.pulsar.theta (float, dataset of group): polar angle of pulsar in radians
    - enterprise.pulsar.Mmat (numpy.array, dataset of group):  ntoa x npar design matrix
    - names_list (list, dataset): List of pulsar names. Will be used to read from hdf5
    
    Args:
        ePsrs (list): list of enterprise.pulsar objects
        noise (dict): dictionary containing all noise flags
        edir (str, optional): directory to store hdf5 into. Must include name of file. Defaults to 'Home/enterprise_psrs.hdf5'.

    Returns:
        edir (str): directory in which file is stored under
    """
    Tspan = hsen.get_Tspan(ePsrs)
    
    #required attributes from enterprise.pulsar objects
    req_attrs = ['toas', 'toaerrs', 'phi', 'theta', 'Mmat', 'pdist']
    
    failed_psrs = []
    for ePsr in ePsrs:
        for attr in req_attrs:
            if not hasattr(ePsr, attr):
                failed_psrs.append((ePsr, attr))
    
    if len(failed_psrs) != 0:
        raise Exception(f'The following enterprise.pulsars do not have the required attributes:\n {[failed_attr for failed_attr in failed_psrs]}')
     
    else:
        name_list = []
        with h5py.File(edir, 'w') as f:
            f.create_dataset('Tspan', (1,), float, data=Tspan)
            for psr in ePsrs:
                #need to add WN covariance matrix here because it needs flags from enterprise.pulsar objects
                psr.N = make_corr(psr, noise)[::thin, ::thin]
                name_list.append(psr.name)
                hdf5_psr = f.create_group(psr.name)
                hdf5_psr.create_dataset('toas', psr.toas.shape, psr.toas.dtype, data=psr.toas)
                hdf5_psr.create_dataset('toaerrs', psr.toaerrs.shape,psr.toaerrs.dtype, data=psr.toaerrs)
                hdf5_psr.create_dataset('phi', (1,), float, data=psr.phi)
                hdf5_psr.create_dataset('theta', (1,), float, data=psr.theta)
                hdf5_psr.create_dataset('designmatrix', psr.Mmat.shape, psr.Mmat.dtype, data=psr.Mmat)
                hdf5_psr.create_dataset('N', psr.N.shape, psr.N.dtype, data=psr.N)
                hdf5_psr.create_dataset('pdist', (2,), float, data=psr.pdist)
                f.flush()
        
            f.create_dataset('names',data=np.array(name_list, dtype=h5py.string_dtype(encoding='utf-8')))
            f.flush()
        return edir

    


if __name__ == '__main__':
    #thinning for noise covariance matrix because N_toa x N_toa size
    thin = 2
    #profile stuff for getting when run time starts
    null_time = time.time()
    Null_time_file.write(f'{null_time}\n')
    Null_time_file.flush()
    
   
   
    ###################################################
    #lists for plotting sensitivity curves
    spectra_list = []
    spectra_list_r = []
    #names list is list of names of pulsars stored within HDF5 file.
    #This list is preferred because it does not require a new creation of enterprise objects
    names_list = [] 
    fyr = 1/(365.25*24*3600)
    

    #read time data for profiling
    with cProfile.Profile() as pr:

        #EITHER SELECT 11 yr or 12 yr
        #pars, tims, noise, rn_psrs, edir, ephem = yr_11_data()
        pars, tims, noise, rn_psrs, edir, ephem = yr_12_data()
        #creates list of enterprise pulars
        ePsrs = hsen.enterprise_creation(pars, tims, ephem)

        #writes enterprise pulsars to hdf5 file
        edir = ent_psr_to_hdf5(ePsrs, noise, edir)
        #list of enterprise pulsars are no longer needed
        del ePsrs

        #opening hdf5 file containing enterprise pulsar attributes
        with h5py.File(edir, 'r') as f:

            #reading Tspan and name of pulsar from h5 file
            Tspan = f['Tspan'][:][0]
            names = f['names'][:]
            #converting byte strings to strings. This is how I could write list of strings to hdf5
            for i in range(len(names)):
                names_list.append(names[i].decode('utf-8'))
            del names

            #parameters used in spectrum noise analysis
            freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),200)
            freqs_gw = freqs
            A_gw = 9e-15
            gamma_gw = 13/3

            total_start_OG = time.time()
            #for-loop to iterate over every pulsar. Original Method
            for name in names_list:
                #reading enterprise pulsar group from h5 file
                h5ePsr = f[name]
                #creating temporary object to store needed attributes. This format is easiest for storing data
                fake_ePsr = PseudoPulsar(name = name, toas = h5ePsr['toas'][:], toaerrs=h5ePsr['toaerrs'][:], phi = h5ePsr['phi'][:][0],
                                theta = h5ePsr['theta'][:][0], pdist=h5ePsr['pdist'][:], N = h5ePsr['N'][:], designmatrix = h5ePsr['designmatrix'][:])
                
                #creates hasasia.pulsar object, writes it to new h5 file with returned directories. Note two different types due to difference
                og_dir = hsen_creation_to_hdf5(fake_ePsr)
                del fake_ePsr

            spectra_list = hsen_spectra_creation(freqs, names_list, path=og_dir)
            ng11yr_sc = hsen.GWBSensitivityCurve(spectra_list)
            ng11yr_dsc = hsen.DeterSensitivityCurve(spectra_list, A_GWB=A_gw)
            sc_hc = ng11yr_sc.h_c
            sc_freqs = ng11yr_sc.freqs
            dsc_hc = ng11yr_dsc.h_c
            dsc_freqs = ng11yr_dsc.freqs
            del ng11yr_sc, ng11yr_dsc, spectra_list

            total_end_OG = time.time()
            Null_time_file.write(f'Total OG: {total_end_OG-total_start_OG}\n')
            Null_time_file.flush()

            total_start_RRF = time.time()
            #for-loop to iterate over every pulsar. Rank-Reduced Method
            for name in names_list:
                h5ePsr = f[name]
                fake_ePsr = PseudoPulsar(name = name, toas = h5ePsr['toas'][:], toaerrs=h5ePsr['toaerrs'][:], phi = h5ePsr['phi'][:][0],
                                theta = h5ePsr['theta'][:][0], pdist=h5ePsr['pdist'][:], N = h5ePsr['N'][:], designmatrix = h5ePsr['designmatrix'][:])
                rrf_dir = hsen_creation_to_hdf5_rrf(fake_ePsr)
                del fake_ePsr

            spectra_list_r = hsen_spectra_creation_rrf(freqs=freqs, freqs_gw=freqs, names=names_list, path=rrf_dir)
            ng11yr_rrf_sc = hsen.GWBSensitivityCurve(spectra_list_r)
            ng11yr_rrf_dsc = hsen.DeterSensitivityCurve(spectra_list_r, A_GWB=A_gw)
            rrf_sc_hc = ng11yr_rrf_sc.h_c
            rrf_sc_freqs = ng11yr_rrf_sc.freqs
            rrf_dsc_hc = ng11yr_rrf_dsc.h_c
            rrf_dsc_freqs = ng11yr_rrf_dsc.freqs
            del ng11yr_rrf_sc, ng11yr_rrf_dsc, spectra_list_r

            total_end_RRF = time.time()
            Null_time_file.write(f'Total RRF: {total_end_RRF-total_start_RRF}\n')
            Null_time_file.flush()
            Null_time_file.close()

            
                

       

        
    #saving time profile data collected on entire run
    with open(path + '/test_time.txt', "w") as file:
        stats = pstats.Stats(pr, stream=file)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()
     
        #all off by a scaling factor of sqrt(2) ##FIND THE SOURCE
        plt.loglog(sc_freqs,sc_hc, label='Norm Stoch', ls='--')
        plt.loglog(dsc_freqs,dsc_hc, label='Norm Det', ls='--')
        plt.loglog(rrf_sc_freqs,rrf_sc_hc, label='RRF Stoch', ls=':')
        plt.loglog(rrf_dsc_freqs,rrf_dsc_hc, label='RRF Det', ls=':')
        plt.ylabel('Characteristic Strain, $h_c$')
        plt.title('NANOGrav 12-year Data Set Sensitivity Curve')
        plt.grid(which='both')
        plt.legend()
        plt.savefig(path+'/GWB_h_c.png')
        plt.close()  

  
    



    
