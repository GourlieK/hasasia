#Kyle Gourlie
#8/2/2023
#library imports
import shutil, h5py
import gc
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



#Memory Profile File Locations
path = r'/home/gourliek/Desktop/Profile_Data'
try:
    os.mkdir(path)
except FileExistsError:
    shutil.rmtree(path)
    os.mkdir(path)
#white noise covariance matrix profile file
corr_matrix_mem = open(path + '/corr_matrix_mem.txt','w')
#creation of hasasia pulsar
hsen_psr_mem = open(path + '/psr_hsen_mem.txt','w')
#Rank Reduced Formalism creation of hasasia pulsar
hsen_psr_RRF_mem= open(path + '/psr_hsen_mem_RRF.txt','w')
#
hsen_psr_spec_mem = open(path + '/psr_hsen_spec_mem.txt','w')
#Rank Reduced Formalism creation of hasasia pulsar
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
################################################################################################################
################################################################################################################
################################################################################################################
#Important Functions Created from 11-yr tutorial


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



def get_psrname(file,name_sep='_'):
    """Function that grabs names of pulsars from parameter files
    
    Returns:
        Pulsar name
    """
    return file.split('/')[-1].split(name_sep)[0]






@profile(stream = corr_matrix_mem)
def make_corr(psr):
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

#create white noise covariance matrix from enterprise pulsar 
def white_corr(epsrs):
    """Generates a list of white noise covariance matrices

    Args:
        epsrs (enterprise.pulsar): pulsar 

    Returns:
        wn_list (list): list of white noise corvariance matrices
    """
    wn_list = []
    for ePsr in epsrs:
        wn_list.append(make_corr(ePsr)[::thin,::thin])
    
    return wn_list



def enterprise_entry(ePsrs, edir):
    """Writes enterprise.pulsar objects onto HDF5 file with WN Covariance matrix attributes.

    - ePsrs (list): List of enterprise.pulsar objects
    """
   
    #computes all white noise covariance matrices for all the pulsars, a list of matrices
    wn_corrs = white_corr(ePsrs)
    Tspan = hsen.get_Tspan(ePsrs)
    #adds each white noise covariance matrix as an attribute
    for i in range(len(ePsrs)):
        ePsrs[i].N = wn_corrs[i]
    
    del wn_corrs

    with h5py.File(edir, 'w') as f:
        Tspan_h5 = f.create_dataset('Tspan', (1,), float)
        Tspan_h5[:] = Tspan
        name_list = np.array(['X' for _ in range(kill_count)], dtype=h5py.string_dtype(encoding='utf-8'))
        
        for i in range(len(ePsrs)): 
            hdf5_psr = f.create_group(ePsrs[i].name)
            hdf5_psr.create_dataset('toas', ePsrs[i].toas.shape, ePsrs[i].toas.dtype, data=ePsrs[i].toas)
            hdf5_psr.create_dataset('toaerrs', ePsrs[i].toaerrs.shape,ePsrs[i].toaerrs.dtype, data=ePsrs[i].toaerrs)
            hdf5_psr.create_dataset('phi', (1,), float, data=ePsrs[i].phi)
            hdf5_psr.create_dataset('theta', (1,), float, data=ePsrs[i].theta)
            hdf5_psr.create_dataset('designmatrix', ePsrs[i].Mmat.shape, ePsrs[i].Mmat.dtype, data=ePsrs[i].Mmat)
            hdf5_psr.create_dataset('N', ePsrs[i].N.shape, ePsrs[i].N.dtype, data=ePsrs[i].N)
            hdf5_psr.create_dataset('pdist', (2,), float, data=ePsrs[i].pdist)
            name_list[i] = ePsrs[i].name
            f.flush()
            
        f.create_dataset('names',data =name_list)
        del ePsrs[:]
        f.flush()
        gc.collect()



def hsen_pulsar_entry(psr, type__, dataset):
    """Writes Hasasia Pulsar attributes to an HDF5 file

    Args:
    - psr (Hasasia Pulsar): Hasasia Pulsar object
    - type__ (str): 'og' XOR 'rrf'
    """
    if type__ == 'og':
        path = r'/home/gourliek/'+str(dataset)+r'_yr_pulsars_og.hdf5'

    elif type__ == 'rrf':
        path = r'/home/gourliek/'+str(dataset)+r'_yr_pulsars_rrf.hdf5'
        
    with h5py.File(path, 'a') as f:
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



@profile(stream=hsen_psr_mem)
def hsen_creation(ePsr, freqs):
    """creation of hasasia pulsar, then saves result into HDF5 file.
    Note that this uses a noise covariance matrix with both red noise and white noise contributions for self.N

    Args:
        ePsr (PseudoPulsar): PseudoPulsar which is read from HDF5 file that save Enterprise.Pulsar attributes
        freqs (numpy.array): frequencies
    """
    #benchmark stuff
    
    corr_from_psd_mem.write(f'Pulsar: {ePsr.name}\n')
    corr_from_psd_mem.flush()
    #start_time = time.time()
    #building red noise powerlaw using standard amplitude and gamma
    plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)

    #if red noise parameters for an individual pulsar is present, add it to standard red noise
    if ePsr.name in rn_psrs.keys():
        Amp, gam = rn_psrs[ePsr.name]
        plaw += hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)  #was +=

    #adding red noise components to the noise covariance matrix via power spectral density
    ePsr.N += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                toas=ePsr.toas[::thin])   
    
    #creating hasasia pulsar tobject 
    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta, 
                        N=ePsr.N, designmatrix=ePsr.Mmat[::thin,:], pdist=ePsr.pdist)
   
    #setting name of hasasia pulsar
    psr.name = ePsr.name
    _ = psr.K_inv
    hsen_pulsar_entry(psr, 'og', dataset)

    #enterprise pulsar is no longer needed
    del ePsr
    print(f"Hasasia Pulsar {psr.name} created\n")


@profile(stream=hsen_psr_RRF_mem)
def hsen_creation_rrf(ePsr):
    """creation of hasasia pulsar, then saves result into HDF5 file. 
    Note that RRF uses only a white noise covariance matrix for self.N

    Args:
        ePsr (PseudoPulsar): PseudoPulsar which is read from HDF5 file that save Enterprise.Pulsar attributes
        freqs (numpy.array): frequencies
    """
    #benchmark stuff
    #creating hasasia pulsar tobject 
    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta, 
                        N=ePsr.N, designmatrix=ePsr.Mmat[::thin,:], pdist=ePsr.pdist)
    
    #setting name of hasasia pulsar
    psr.name = ePsr.name
    _ = psr.K_inv
    hsen_pulsar_entry(psr, 'rrf', dataset) 

    #enterprise pulsar is no longer needed
    del ePsr
    print(f"Hasasia RRF Pulsar {psr.name} created\n")


@profile(stream=hsen_psr_spec_mem)
def hsen_spectra_creation(freqs, names, dataset)->list:
    """Creation of hasasia spectrum pulsars 

    Args:
        - freqs (numpy.array): frequencies
        - names (list): list containing all names of pulsars

    Returns:
        - spectras (list): list of Hasasia Spectrum Pulsars
    """
    
    spectras = []
    path = r'/home/gourliek/'+str(dataset)+r'_yr_pulsars_og.hdf5'
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
def hsen_spectra_creation_rrf(freqs, freqs_gw, names, dataset)->list:
    """Creation of hasasia spectrum pulsars using RRF

    Args:
        - freqs (numpy.array): frequencies
        - names (list): list containing all names of pulsars

    Returns:
        - spectras (list): list of Hasasia Spectrum Pulsars
    """
    path = r'/home/gourliek/'+str(dataset)+r'_yr_pulsars_rrf.hdf5'
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
                spec_psr = hsen.RRF_Spectrum(pseudo, freqs_gw=freqs_gw,amp = Amp, gamma = gam, freqs=freqs)
                spec_psr.name = pseudo.name
            
            else:
                #creates spectrum hasasia pulsar to calculate characteristic straing
                spec_psr = hsen.RRF_Spectrum(pseudo, freqs=freqs, freqs_gw=freqs)
                spec_psr.name = pseudo.name

            print(f'Hasasia Spectrum RRF {spec_psr.name} created\n')
            #hasasia pulsar no longer needed
            del pseudo
            ########################################
            #quantity is called up so it is computed
            get_NcalInv_RFF_mem.write(f'Pulsar: {name}\n')
            get_NcalInv_RFF_mem.flush()
            NcalInv_time_start = time.time()
            _ = spec_psr.NcalInv
            NcalInv_time_end = time.time()
            end_time = time.time()
            ########################################
            NcalInv_RRF_time_file.write(f"RRF{name}\t{NcalInv_time_end-NcalInv_time_start}\n")
            time_increments.write(f"RRF {name} {start_time-null_time} {end_time-null_time}\n")
            NcalInv_RRF_time_file.flush()
            time_increments.flush()
            spectras.append(spec_psr)
    return spectras







"""
@profile(stream=sens_mem_RRF)
def rrf_array_construction(ePsr, freqs):
    
    #benchmark stuff
    
    hc_time_start = time.time()
    start_time = time.time()
        
    #creating hasasia pulsar tobject 
    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta, 
                        N=ePsr.N, designmatrix=ePsr.Mmat[::thin,:])
    
    
    #setting name of hasasia pulsar
    psr.name = ePsr.name
    #hsen_pulsar_entry(psr, 'og')    _
    #extra variable needed for benchmarking

    #enterprise pulsar is no longer needed
    del ePsr
    print(f"Hasasia Pulsar {psr.name} created\n")
    #spectra_entry(psr)
    hc_time_start = time.time()
    
    #if red noise parameters for an individual pulsar is present, add it to standard red noise
    if psr.name in rn_psrs.keys():
        Amp, gam = rn_psrs[psr.name]
        #creates spectrum hasasia pulsar to calculate characteristic straing
        spec_psr = hsen.RRF_Spectrum(psr, amp = Amp, gamma = gam, freqs=freqs)
        spec_psr.name = psr.name
    
    else:
        #creates spectrum hasasia pulsar to calculate characteristic straing
        spec_psr = hsen.RRF_Spectrum(psr, freqs=freqs)
        spec_psr.name = psr.name
    
    #hasasia pulsar no longer needed
    del psr

    hc_time_end = time.time()
    name = spec_psr.name
   
    hc_RRF_time_file.write(f"RRF{name}\t{hc_time_end-hc_time_start}\n")

    print(f"Hasasia Spectrum RRF Pulsar {name} created\n")

    #benchmark stuff
    end_time = time.time()
    time_increments.write(f" RRF{name} {start_time-null_time} {end_time-null_time}\n")
    print('\rPSR RRF {0} complete\n'.format(name),end='',flush=True)
    
    return spec_psr



@profile(stream=sens_mem)
def array_construction(ePsr, freqs):

    #benchmark stuff
    get_NcalInv_mem.write(f'Pulsar: {ePsr.name}\n')
    corr_from_psd_mem.write(f'Pulsar: {ePsr.name}\n')
    start_time = time.time()
    #building red noise powerlaw using standard amplitude and gamma
    plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)

    #if red noise parameters for an individual pulsar is present, add it to standard red noise
    if ePsr.name in rn_psrs.keys():
        Amp, gam = rn_psrs[ePsr.name]
        plaw += hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)  #was +=

    #adding red noise components to the noise covariance matrix via power spectral density
    ePsr.N += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                toas=ePsr.toas[::thin])   
    
    #creating hasasia pulsar tobject 
    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                        toaerrs=ePsr.toaerrs[::thin],
                        phi=ePsr.phi,theta=ePsr.theta, 
                        N=ePsr.N, designmatrix=ePsr.Mmat[::thin,:])
    
    #setting name of hasasia pulsar
    psr.name = ePsr.name
    #hsen_pulsar_entry(psr, 'og')

    #enterprise pulsar is no longer needed
    del ePsr
    print(f"Hasasia Pulsar {psr.name} created\n")

    #creates spectrum hasasia pulsar to calculate characteristic strain
    time_hc_start = time.time()
    spec_psr = hsen.Spectrum(psr, freqs=freqs)
    spec_psr.name = psr.name

    #hasasia pulsar no longer needed
    del psr
    _ = spec_psr.NcalInv
   
    time_hc_end = time.time()
    hc_time_file.write(f"{spec_psr.name}\t{time_hc_end - time_hc_start}\n")

    name = spec_psr.name

    print(f"Hasasia Spectrum Pulsar {name} created\n")

    #benchmark stuff
    end_time = time.time()
    time_increments.write(f"{name} {start_time-null_time} {end_time-null_time}\n")
    print('\rPSR {0} complete\n'.format(name),end='',flush=True)

    return spec_psr
"""      




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
    pardir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/'
    timdir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/'
    noise_dir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master'
    noise_dir += '/nano11y_data/noisefiles/'
    psr_list_dir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/psrlist.txt'

    #directory name of enterprise hdf5 file
    dataset=11
    

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

    edir = '/home/gourliek/11_yr_enterprise_pulsars.hdf5'
    dataset=11
    
    
      
    return pars, tims, noise, rn_psrs, edir, dataset



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

    edir = '/home/gourliek/12_yr_enterprise_pulsars.hdf5'
    dataset = 12
    
    return pars, tims, noise, rn_psrs, edir, dataset

    
    
def enterprise_creation(pars, tims, dataset):
    #generating enterprise pulsars with counter to choose how many i want
    enterprise_Psrs = []
    count = 1
    if dataset == 11:
        ephem = 'DE436'
    elif dataset == 12:
        ephem = 'DE438'
    
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



def hasasia_write(file, psr_names):
    """Function that writes hasasia pulsars to HDF5. Comment this out if HDF5 files already created"""
    for name in psr_names:
        psr = file[name]
    
        pseudo = PseudoPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                theta = psr['theta'][:][0], pdist=psr['pdist'][:], N=psr['N'][:])
        pseudo.name = name
        pseudo.Mmat= psr['designmatrix'][:]

        #converts each pseduo into hasasia pulsar, then writes it to disk
        
        hsen_psr_mem.write(f'Pulsar: {name}\n')
        hsen_psr_RRF_mem.write(f'Pulsar: {name}\n')
        hsen_psr_mem.flush()
        hsen_psr_RRF_mem.flush()
        
        hsen_creation(pseudo, freqs)
        hsen_creation_rrf(pseudo)
        del pseudo



  
 
################################################################################################################
################################################################################################################
################################################################################################################
if __name__ == '__main__':
    null_time = time.time()
    Null_time_file.write(f'{null_time}\n')
    Null_time_file.flush()
    Null_time_file.close()
    ###################################################
    #max is 34 for 11yr dataset
    #max is 45 for 12yr dataset
    kill_count =  3
    thin = 1
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
        pars, tims, noise, rn_psrs, edir, dataset = yr_11_data()
        #pars, tims, noise, rn_psrs, edir, dataset = yr_12_data()
        #ePsrs = enterprise_creation(pars, tims, dataset)
        #exit()

        

        #frequencies used for PSD analysis
        
        ##################################################
        #IF you need to re-create enterprise pulsars
        #enterprise_entry(ePsrs, edir)
        #exit()
        ##################################################
        
        with h5py.File(edir, 'r') as f:
            #memory map to hdf5 file
            Tspan_pt = f['Tspan']
            names_pt = f['names']

            #disk to memory
            Tspan = Tspan_pt[:][0]
            names = names_pt[:]

            freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),200)

            #converting byte strings to strings. This is how I could write list of strings to hdf5
            for i in range(kill_count):
                names_list.append(names[i].decode('utf-8'))

            del names

            #this will loop through every pulsar created from HDF5 file
            #hasasia_write(f, names_list)
            spectra_list_r = hsen_spectra_creation_rrf(freqs=freqs, freqs_gw=freqs, names=names_list, dataset=dataset)
            ng11yr_rrf_sc = hsen.GWBSensitivityCurve(spectra_list_r)
            ng11yr_rrf_dsc = hsen.DeterSensitivityCurve(spectra_list_r)
            rrf_sc_hc = ng11yr_rrf_sc.h_c
            rrf_sc_freqs = ng11yr_rrf_sc.freqs
            rrf_dsc_hc = ng11yr_rrf_dsc.h_c
            rrf_dsc_freqs = ng11yr_rrf_dsc.freqs
            del ng11yr_rrf_sc, ng11yr_rrf_dsc, spectra_list_r

            spectra_list = hsen_spectra_creation(freqs, names_list, dataset)
            ng11yr_sc = hsen.GWBSensitivityCurve(spectra_list)
            ng11yr_dsc = hsen.DeterSensitivityCurve(spectra_list)
            sc_hc = ng11yr_sc.h_c
            sc_freqs = ng11yr_sc.freqs
            dsc_hc = ng11yr_dsc.h_c
            dsc_freqs = ng11yr_dsc.freqs
            del ng11yr_sc, ng11yr_dsc, spectra_list
                

       

        
    #saving time profile data collected on entire run
    with open(path + '/test_time.txt', "w") as file:
        stats = pstats.Stats(pr, stream=file)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()
     
        #all off by a scaling factor of sqrt(2) ##FIND THE SOURCE
        plt.loglog(sc_freqs,sc_hc, label='Norm Stoch', c='blue')
        plt.loglog(dsc_freqs,dsc_hc, label='Norm Det', c='red')
        plt.loglog(rrf_sc_freqs,rrf_sc_hc, label='RRF Stoch', linestyle='--')
        plt.loglog(rrf_dsc_freqs,rrf_dsc_hc, label='RRF Det', linestyle='--')
        plt.ylabel('Characteristic Strain, $h_c$')
        plt.title('NANOGrav 12-year Data Set Sensitivity Curve')
        plt.grid(which='both')
        plt.legend()
        plt.savefig(path+'/GWB_h_c.png')
        plt.close()


  
    



    
