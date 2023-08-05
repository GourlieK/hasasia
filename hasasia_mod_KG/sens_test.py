import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import glob, pickle, json, cProfile, pstats
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky
import matplotlib as mpl
import healpy as hp
import astropy.units as u
import astropy.constants as c
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [5,3]
mpl.rcParams['text.usetex'] = True
from enterprise.pulsar import Pulsar as ePulsar





#finds the name of a pulsar based from filename
def get_psrname(file,name_sep='_'):
    return file.split('/')[-1].split(name_sep)[0]





def data_organization():

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



    #list of types of whitenoise
    w_n_key_label_list = [] 

    #White noise is now a list of dictionaries with each dictionary having a pulsar name as the key
    #The values that the key unlock are various white noise parameters dictionary where the key to
    #that dictionary is the type of parameter it is
    for i in range(num):
        w_n_params = white_noise[i][psr_name_list[i]]     #list of white noise parameters for each pulsar
        for j in range(len(w_n_params)):
            raw_key = list(w_n_params[j].keys())[0]   #only one key so [0], not [1]
            key_label = raw_key.replace(psr_name_list[i]+'_',"")  #removing pulsar name 
            key_label = key_label.split("_")   
            del key_label[-1]  #removing efac, equad, ecorr label
            #removes all labels with log10 cause its useless. I can take log10 if need so
            if key_label.count('log10') == 1:
                continue

            key_label = "_".join(key_label)    
            w_n_key_label_list.append(key_label) 
            
            #if key == psr_name_list[i]+ '_430_ASP_efac':
                #print(f'430_ASP_efac:{w_n_params[j][key]}') 



                #prints very values
            #if w_n_params[j].keys() == psr_name_list[i]+ '_430_ASP_efac':
                #print(w_n_params[j][psr_name_list[i]+ '_430_ASP_efac'])    #prints every parame

    #efac, equad, ecorr
    w_n_key_label_list = sorted(list(set(w_n_key_label_list)))
    return par_files, tim_files, noise, psr_name_list, red_noise, white_noise, w_n_key_label_list









par_files, tim_files, noise, psr_names, red_noise, white_noise, white_noise_type = data_organization()



#unlocks all values within the dictionary
'''for i in range(num):
    w_n_params = white_noise[i][psr_name_list[i]] 
    for j in range(len(w_n_params)):
        key = list(w_n_params[j].keys())[0] 
        for name in key_label_list:
            name = '_'+name 
            if key == psr_name_list[i] + name:
                print(f"{psr_name_list[i]} {name}: {w_n_params[j][psr_name_list[i]+ name]}")'''








    
        








#To access red_noise values, use for-loop of psr_name_list, and use the name from psr_name_list as key to values
#for i in range(num):
    #log_10_A_value = red_noise[i][psr_name_list[i]][0]   #enters list, accesses key, and chooses first value in tuple
    #gamma_value = red_noise[i][psr_name_list[i]][1]


#removing no longer needed lists
#del log_10_A__label_list, gamma_label_list, par_name_list, tim_name_list
   



psrs = []
counter = 1
for p, t in zip(par_files, tim_files):
    psr = ePulsar(p, t, ephem='DE438')
    psrs.append(psr)
    print('\rPSR {0} complete\n'.format(psr.name),end='',flush=True)
    counter +=1





def make_corr(psr):
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas,psr.toaerrs,
                                         flags=psr.flags['f'],dt=1)

    #finding the types of white noise
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
    print("t_9")   
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)]
    print("t_10")  
    J = sl.block_diag(*j)
    print("t_11")  

    #ISSUE HERE WHEN RUNNING J1713
    corr = np.diag(sigma_sqr) + J
    print("t_12")  
    return corr





 #increase frequencies to crash pc


Tspan = hsen.get_Tspan(psrs)
fyr = 1/(365.25*24*3600)
freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),300)





thin = 5   #set to 1 because correlation matrix constructed from the red spectral density function won't match the matrix size of the white noise correlation matrix

count = 1
kill_num = 5
for i in range(len(psr_names)):
    ePsr = psrs[i]
    print(count, ePsr.name, "test 1")

    if count >= kill_num:   
        continue
    else:

        if ePsr.name == 'J1713+0747':
            continue

        elif ePsr.name == 'J1747-4036':
            continue

        elif ePsr.name == 'J1853+1303':
            continue


        elif ePsr.name == 'J1903-0327':
            continue

        
    
        else:
        
            #it is dying here
            corr = make_corr(ePsr)#[::thin,::thin]
            print(count, ePsr.name, "test 2")
            A = 10**(red_noise[i][psr_names[i]][0])
            gamma=red_noise[i][psr_names[i]][1]
            plaw = hsen.red_noise_powerlaw(A=A, gamma=gamma, freqs=freqs)
            corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                                        toas=ePsr.toas[::thin])
            psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                                toaerrs=ePsr.toaerrs[::thin],
                                phi=ePsr.phi,theta=ePsr.theta, 
                                N=corr, designmatrix=ePsr.Mmat[::thin,:])

            psr.name = ePsr.name
            psrs.append(psr)
            del ePsr
            print('\rPSR {0} complete'.format(psr.name),end='',flush=True)

            count +=1



specs = []
for p in psrs:
    sp = hsen.Spectrum(p, freqs=freqs)
    _ = sp.NcalInv
    specs.append(sp)
    print('\rPSR {0} complete'.format(p.name),end='',flush=True)



fig=plt.figure(figsize=[15,45])
j = 1

for sp,p in zip(specs,psrs):
    fig.add_subplot(12,3,j)
    a = sp.h_c[0]/2*1e-14
    alp = -3/2
    plt.loglog(sp.freqs[:150],a*(sp.freqs[:150])**(alp),
                color='C1',label=r'$f^{-3/2}$')
    plt.ylim(2e-15,2e-10)
    plt.loglog(sp.freqs,sp.h_c, color='C0')
    plt.rc('text', usetex=True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Characteristic Strain, $h_c$')
    plt.legend(loc='upper left')
    plt.title(p.name)
    j+=1
fig.tight_layout()
plt.show()
plt.close()


