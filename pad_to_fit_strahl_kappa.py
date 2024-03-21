import urllib.request
import cdflib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
import time
import datetime
import math
import pandas as pd
from pandas import DataFrame
import time
import cmocean
from os import listdir
from os.path import isfile, join
from scipy import stats
from scipy import integrate
from numpy import linalg as la
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.ndimage import map_coordinates
import scipy.special
from scipy.special import gamma
from mpl_toolkits import mplot3d
from pynverse import inversefunc
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from datetime import datetime
import matplotlib.gridspec as gridspec
from itertools import groupby
from scipy import stats as st
from fit_functions import *
from math import floor, log10
from num2tex import num2tex
import lmfit
from lmfit import fit_report
import pickle
from pathlib import Path

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
mpl.rcParams['axes.linewidth'] = 1.3
mpl.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts}'})
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.linewidth'] = 1.3
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['xtick.minor.pad'] = 2
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.major.width'] = 120
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['ytick.major.pad'] = 2
mpl.rcParams['ytick.minor.pad'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams.update({'figure.max_open_warning': 0})
s_set = 2
lw_set = 0.8

### constants #
m_e=9.109*10.**(-28.)
eV_to_J=1.602*(10.**(-19.))
eV_to_Erg = 1.602*(10.**(-12.))
m0=1.2566*10.**(-6.)
kb=1.3806*10.**(-23.)
m_p = 1.6726*10.**(-24)


def sh(x): return print(np.shape(x))

def pt(obj):
    namespace = globals()
    return print([name for name in namespace if namespace[name] is obj][0]+' =',obj)

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def make_df(path, data, name) :
    DataFrame(data).to_pickle(path+name)

def find_nearest(array,value):
    idx = int(np.searchsorted(array, value, side="left"))
    #searchsorted faster than where - uses an arranged array
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return int(idx-1)
    else:
        return int(idx)

def eV_to_vel(input) :
    return np.sqrt((2*input*eV_to_Erg)/m_e)

def vel_to_eV(input) :
    return (m_e/2)*(input**2)*(1/eV_to_Erg)

def vel_to_eV_p(input) :
    return (m_p/2)*(input**2)*(1/eV_to_Erg)

def per_iteration(pars, iteration, resid, *args, **kws):
    print(" ITER ", iteration, [f"{p.name} = {p.value:.5f}" for p in pars.values()])

def round_to_n(x, n) :
    if np.sign(x) == 1 :
        return np.round(x, -int(floor(log10(x))) + (n - 1))
    if np.sign(x) == -1 :
        return -1*np.round(np.abs(x), -int(floor(log10(np.abs(x)))) + (n - 1))

''' integrates VDFs respecting the coordiante system of EAS instrument and the way lmfit produces .eval functions '''
def int_vdf(f, v_par, v_perp) :
    int_temp = []
    for i in range(len(v_par)) :
        int_temp.append(np.trapz(np.flipud(v_perp)*np.flipud(f[:,i]), np.flipud(v_perp)))
    return 2*np.pi*np.trapz(np.array(int_temp), v_par)

''' Paths '''
# gen_path = '/Users/jc/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research/Spacecrafts/solar orbiter/make_data_products/EAS/'
gen_path = '/Users/jessecoburn/Documents/Research/Spacecrafts/solar orbiter/make_data_products/EAS/'
plot_path = gen_path+'plots/pad_to_fit/'
data_save_path = '/Volumes/My Passport for Mac/Solar Orbiter/SWA/EAS/PAD_FIT/'

''' data dumped for quicker run tiems '''
dump_path = '/Volumes/My Passport for Mac/Solar Orbiter/SWA/EAS/PAD_FIT_dump/'

''' data reading from master files'''
data_path = '/Volumes/My Passport for Mac/Solar Orbiter/'
''' using LPP corrected data '''
EAS_PAD_save_path = data_path+'SWA/EAS/PAD_LPP_corrected/'
EAS_DATA_save_path = data_path+'SWA/EAS/DATA_LPP_corrected/'
''' using non-LPP corrected data '''
# EAS_PAD_save_path = data_path+'SWA/EAS/PAD/'
# EAS_DATA_save_path = data_path+'SWA/EAS/DATA/'

RPW_save_path = data_path+'RPW/DATA/SC_DENS/'
PAS_save_path = data_path+'SWA/PAS/DATA/'

# file_name = '20220226T000008-20220226T060508'
file_name = '20220302T180509-20220302T235949'
date = '20220302'

ang_name = 'PAD_'
pa_vec_name = 'data_set_'
mag_name = 'B_avg_srf_'
epoch_name = 'PAD_epoch_'
time_name = 'SWA_EAS1_Time_'
pitch_angle_name = 'pitch_angles_arr_'

''' start of large comment '''

# ''' Read SWA-EAS data '''
# pad_in = np.array(pd.read_pickle(EAS_PAD_save_path+ang_name+file_name))
# pad_in_shape = tuple(np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+ang_name+file_name+'_shape')), axis = 1))
# pad_in = np.reshape(pad_in, pad_in_shape)
#
# ''' pa vec [ time, energy, (azimuth*elevation*2heads), [pa, psd, v_par, v_perp] ] '''
# pa_vec = np.array(pd.read_pickle(EAS_PAD_save_path+pa_vec_name+file_name))
# pa_vec_shape = tuple(np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+pa_vec_name+'shape_'+file_name)), axis = 1))
# pa_vec = np.reshape(pa_vec, pa_vec_shape)
# B_avg = np.array(pd.read_pickle(EAS_PAD_save_path+mag_name+file_name))
# epoch_EAS = np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+epoch_name+file_name)), axis = 1)
# time_EAS = list(np.array(pd.read_pickle(EAS_DATA_save_path+time_name+file_name)))
#
# pitch_angles = np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+pitch_angle_name+file_name)), axis = 1) #edges of the pitch angle bins
# swa_energy = np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+'swa_energy_'+file_name)), axis = 1) #energy bin centers
# swa_energy_delta_lower = np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+'swa_energy_delta_lower_'+file_name)), axis = 1) #energy low_cer edge
# swa_energy_delta_upper = np.sum(np.array(pd.read_pickle(EAS_PAD_save_path+'swa_energy_delta_upper_'+file_name)), axis = 1) #energy upper edge
#
# ''' Read RPW density data '''
# epoch_RPW = np.sum(np.array(pd.read_pickle(RPW_save_path+'epoch_'+date)), axis = 1)
# time_RPW = np.array(pd.read_pickle(RPW_save_path+'time_'+date))
# q_flag_n_sc = np.array(pd.read_pickle(RPW_save_path+'q_flag_'+date))
# n_sc = np.array(pd.read_pickle(RPW_save_path+'n_sc_'+date))
#
# ''' read in PAS SW velocity data '''
# epoch_PAS = np.sum(np.array(pd.read_pickle(PAS_save_path+'epoch_'+date)), axis = 1)
# n = np.sum(np.array(pd.read_pickle(PAS_save_path+'n_'+date)), axis = 1)
# u = np.array(pd.read_pickle(PAS_save_path+'u_SRF_'+date))
# T = np.array(pd.read_pickle(PAS_save_path+'T_'+date))
#
# EAS_time_interval = 1*(10**9) # 1 second in nanoseconds
# def interp_dens_to_EAS(n_sc, n, epoch_RPW, epoch_PAS, epoch_EAS) :
#
#     n_sc_EAS, n_EAS, u_EAS, T_EAS = [], [], [], []
#     for i in range(len(epoch_EAS)):
#
#         beg_PAS_idx, end_PAS_idx = find_nearest(epoch_PAS, epoch_EAS[i] - EAS_time_interval/2), find_nearest(epoch_PAS, epoch_EAS[i] + EAS_time_interval/2)
#         n_EAS.append(np.mean(n[beg_PAS_idx:end_PAS_idx+1]))
#         u_EAS.append(np.array([np.mean(u[beg_PAS_idx:end_PAS_idx+1, 0]), np.mean(u[beg_PAS_idx:end_PAS_idx+1, 1]), np.mean(u[beg_PAS_idx:end_PAS_idx+1, 2])]))
#         T_EAS.append(np.mean(T[beg_PAS_idx:end_PAS_idx+1]))
#
#         beg_RPW_idx, end_RPW_idx = find_nearest(epoch_RPW, epoch_EAS[i] - EAS_time_interval/2), find_nearest(epoch_RPW, epoch_EAS[i] + EAS_time_interval/2)
#         n_sc_EAS.append(np.mean(n_sc[beg_RPW_idx:end_RPW_idx+1]))
#
#     return np.array(n_EAS), np.array(n_sc_EAS), np.array(u_EAS), np.array(T_EAS)
#
# n_EAS, n_sc_EAS, u_EAS, T_EAS = interp_dens_to_EAS(n_sc, n, epoch_RPW, epoch_PAS, epoch_EAS)
#
# u_EAS = np.array(pd.read_pickle(EAS_PAD_save_path+'u_avg_srf_'+file_name))
#
# ''' dump smaller data set '''
# # beg, end = 0, len(epoch_EAS)
# # print(beg, end)
# beg, end = 80, 90
#
# make_df(dump_path, np.ravel(pad_in[beg:end,:,:,:]), 'pad_in')
# make_df(dump_path, pad_in[beg:end,:,:,:].shape, 'pad_in_shape')
#
# make_df(dump_path, np.ravel(pa_vec[beg:end,:,:,:]), 'pa_vec')
# make_df(dump_path, pa_vec[beg:end,:,:,:].shape, 'pa_vec_shape')
#
# make_df(dump_path, B_avg[beg:end,:], 'B_avg')
# make_df(dump_path, epoch_EAS[beg:end], 'epoch_EAS')
# make_df(dump_path, time_EAS[beg:end], 'time_EAS')
#
# make_df(dump_path, pitch_angles, 'pitch_angles')
# make_df(dump_path, swa_energy, 'swa_energy')
# make_df(dump_path, swa_energy_delta_lower, 'swa_energy_delta_lower')
# make_df(dump_path, swa_energy_delta_upper, 'swa_energy_delta_upper')
#
# make_df(dump_path, n_EAS[beg:end], 'n_EAS')
# make_df(dump_path, T_EAS[beg:end], 'T_EAS')
# make_df(dump_path, n_sc_EAS[beg:end], 'n_sc_EAS')
# make_df(dump_path, u_EAS[beg:end,:], 'u_EAS')

''' end of large comment '''

''' Read smaller SWA-EAS data set '''
pad_in = np.array(pd.read_pickle(dump_path+'pad_in'))
pad_in_shape = tuple(np.sum(np.array(pd.read_pickle(dump_path+'pad_in_shape')), axis = 1))
pad_in = np.reshape(pad_in, pad_in_shape)

''' notes on data sets'''
''' pad_in (time, energy bin = 64, pa bins = 36, (mean PSD [km^6/s^3], number)),'''

pa_vec_in = np.array(pd.read_pickle(dump_path+'pa_vec'))
pa_vec_in_shape = tuple(np.sum(np.array(pd.read_pickle(dump_path+'pa_vec_shape')), axis = 1))
pa_vec_in = np.reshape(pa_vec_in, pa_vec_in_shape)

''' pa vec [ time, energy bin = 64, (azimuth*elevation*2heads) = 1024, [pa, psd, v_par, v_perp, counts] ] '''
'''  final dimension units [degrees, km^6 / s^3, m/s, m/s, number] ]     '''
''' B_avg is in nT '''

''' MAG B SRF averaged to EAS '''
B_avg = np.array(pd.read_pickle(dump_path+'B_avg'))

epoch_EAS = np.sum(np.array(pd.read_pickle(dump_path+'epoch_EAS')), axis = 1)
time_EAS = list(np.array(pd.read_pickle(dump_path+'time_EAS')))

pitch_angles = np.sum(np.array(pd.read_pickle(dump_path+'pitch_angles')), axis = 1) #edges of the pitch angle bins
swa_energy = np.sum(np.array(pd.read_pickle(dump_path+'swa_energy')), axis = 1) #energy bin centers
swa_energy_delta_lower = np.sum(np.array(pd.read_pickle(dump_path+'swa_energy_delta_lower')), axis = 1) #energy lower edge
swa_energy_delta_upper = np.sum(np.array(pd.read_pickle(dump_path+'swa_energy_delta_upper')), axis = 1) #energy upper edge

n_sc = np.sum(np.array(pd.read_pickle(dump_path+'n_sc_EAS')), axis = 1) # RPW's electron desnity averaged to EAS time
n_p = np.sum(np.array(pd.read_pickle(dump_path+'n_EAS')), axis = 1) # proton density from PAS, averaged to EAS time.
u = np.array(pd.read_pickle(dump_path+'u_EAS')) # proton vleocity in SRF at EAS time
''' n, n_sc, u are cm^{-3}, cm^{-3}, km/s '''

''' 1-particle noise '''
one_particle_noise_shape = tuple(np.sum(np.array(pd.read_pickle(dump_path+'one_particle_noise_shape')), axis = 1))
one_particle_noise = np.reshape(np.sum(np.array(pd.read_pickle(dump_path+'one_particle_noise')), axis = 1), one_particle_noise_shape)
# one_particle_noise = one_particle_noise[beg_pad_idx:end_pad_idx,:,ene_upp_idx:,:]
one_particle_noise = np.mean(one_particle_noise, axis = 0)
one_particle_noise = np.mean(one_particle_noise, axis = 0)
one_particle_noise = np.mean(one_particle_noise, axis = 1)

''' convert everything to CGS units '''
psd_to_cgs = 10**(-30)
m_to_cm = 10**2
km_to_cm = 10**5
pad_in[:,:,:,0] = pad_in[:,:,:,0]*psd_to_cgs
pa_vec_in[:,:,:,1] = pa_vec_in[:,:,:,1]*psd_to_cgs
pa_vec_in[:,:,:,2] = pa_vec_in[:,:,:,2]*m_to_cm
pa_vec_in[:,:,:,3] = pa_vec_in[:,:,:,3]*m_to_cm
u = u*km_to_cm
one_particle_noise = (10**(-14))*one_particle_noise # m^6 to cm^6 * 100

''' -------------------------------------------- '''
'''                 Analysis                     '''
''' -------------------------------------------- '''

''' define fit functions '''
core = lmfit.Model(bi_Max, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp'])

halo = lmfit.Model(kappa_vdf, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa'])

halo_hallow = lmfit.Model(kappa_flattop_vdf, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa'], nan_policy = 'omit')

pad_model = lmfit.Model(pitch_angle_model, independent_vars = ['alpha'], \
    param_names = ['P_B', 'P_0', 'W_0', 'P_180', 'W_180'], nan_policy = 'omit')

paw_model = lmfit.Model(pitch_angle_width_model, independent_vars = ['alpha'], \
    param_names = ['P', 'W'], nan_policy = 'omit')

beam = lmfit.Model(kappa_beam, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa'])

vdf_1beam = lmfit.Model(core_kappa_beam_halo, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c', 'n_b', 'u_par_b', 'v_th_par_b', 'v_th_perp_b', 'kappa_b', 'n_h', 'u_par_h', 'v_th_par_h', 'v_th_perp_h', 'kappa', 'n_sc'])

vdf_2beam = lmfit.Model(core_kappa_beam_beam_halo, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c', 'n_b_par', 'u_par_b_par', 'v_th_par_b_par', 'v_th_perp_b_par', 'kappa_b_par', 'n_b_anti_par', 'u_par_b_anti_par', 'v_th_par_b_anti_par', 'v_th_perp_b_anti_par', 'kappa_b_anti_par', 'n_h', 'u_par_h', 'v_th_par_h', 'v_th_perp_h', 'kappa', 'n_sc'])

''' function used in analysis that are under construction '''
def pad_fit_function(psd_data, alpha_center, alpha_edge) :

    '''param_names = ['P_B', 'P_0', 'W_0', 'P_180', 'W_180']'''

    alpha_center = alpha_center[~np.isnan(psd_data)]
    psd_data = psd_data[~np.isnan(psd_data)]

    pad_const = [[np.nanmean(psd_data)*0.9, None], [None, None], [0, 45], [None, None], [0, 45]]
    p_pad = lmfit.Parameters()
    p_pad.add('P_B', np.nanmean(psd_data), vary = False, min = pad_const[0][0], max = pad_const[0][1])
    p_pad.add('P_0', np.sign(psd_data[0] - np.nanmean(psd_data))*psd_data[0], vary = False, min = pad_const[1][0], max = pad_const[1][1])
    p_pad.add('W_0', 10, vary = False, min = pad_const[2][0], max = pad_const[2][1])
    p_pad.add('P_180', np.sign(psd_data[-1] - np.nanmean(psd_data))*psd_data[-1], vary = False, min = pad_const[3][0], max = pad_const[3][1])
    p_pad.add('W_180', 10, vary = False, min = pad_const[4][0], max = pad_const[4][1])

    ''' method employed here '''
    try :
        r_pad = pad_model.fit(psd_data, p_pad, method = method, alpha = alpha_center, max_nfev=300, nan_policy = 'omit')
    except (ValueError, TypeError) :
        r_pad = None

    def fit_update(p_pad, r_pad, vary_list, psd_data, alpha, pad_const) :
        p_pad.add('P_B', r_pad.params['P_B'].value, vary = vary_list[0], min = pad_const[0][0], max = pad_const[0][1])
        p_pad.add('P_0', r_pad.params['P_0'].value, vary = vary_list[1], min = pad_const[1][0], max = pad_const[1][1])
        p_pad.add('W_0', r_pad.params['W_0'].value, vary = vary_list[2], min = pad_const[2][0], max = pad_const[2][1])
        p_pad.add('P_180', r_pad.params['P_180'].value, vary = vary_list[3], min = pad_const[3][0], max = pad_const[3][1])
        p_pad.add('W_180', r_pad.params['W_180'].value, vary = vary_list[4], min = pad_const[4][0], max = pad_const[4][1])

        try :
            r_pad = pad_model.fit(psd_data, p_pad, method = method, alpha = alpha_center, max_nfev=300, nan_policy = 'omit')
        except (ValueError, TypeError) :
            # print('pad dint fit')

            r_pad = None

        return r_pad

    vary_list = [[True, False, False, False, False], \
    [False, True, False, True, False], \
    [True, False, True, False, True]]#, \
    # [True, True, True, True, True]]

    for i in range(len(vary_list)) :
        if r_pad != None :
            r_pad = fit_update(p_pad, r_pad, vary_list[i], psd_data, alpha_center, pad_const)

    ''' put results back into fit function '''
    if r_pad != None :
        fit_pad = pad_model.eval(r_pad.params, alpha = alpha_edge)
        return r_pad.params, fit_pad
    else:
        # print('dint fit pad model')
        fit_pad = pad_model.eval(p_pad, alpha = alpha_edge)
        return p_pad, fit_pad

def pad_fit_operation() :

    pad_params_energy, pad_fit_energy = [], []
    for j in range(len(swa_energy)) :
        data_in = pad[j,:,0]
        mask = (data_in != 0.0)
        if len(data_in[mask]) < 4 : # essentially if there were no points.
            temp = [np.nan, np.nan, np.nan, np.nan, np.nan]
            temp_fit_pad = np.full(len(pitch_angles), np.nan)
        else :
            temp_params, temp_fit_pad = pad_fit_function(data_in[mask], pitch_angle_centers[mask], pitch_angles)
            temp = [temp_params['W_0'].value, temp_params['W_180'].value, temp_params['P_0'].value, temp_params['P_180'].value, temp_params['P_B'].value]
        pad_params_energy.append(np.array(temp))
        pad_fit_energy.append(np.array(temp_fit_pad))

    pad_params_energy = np.array(pad_params_energy)
    pad_fit_energy = np.array(pad_fit_energy)

    return np.array(pad_params_energy), np.array(pad_fit_energy)

def pad_fit_plot() :

    ''' plot results '''
    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    color_lines = color_lines(np.linspace(0, 1, 4 + 1))

    fig, ax = plt.subplots(3, 1, figsize = (5.5,11), constrained_layout=True)
    # fig.subplots_adjust(hspace = 0.1, wspace = 0.1)


    ''' Pitch angle fit '''
    ax[0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', ms = s_set, elinewidth = 0.7, capsize = 1, color = color_lines[0], label = 'VDF averaged ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0].set_xlabel('Pitch angle [deg.]')

    ax[0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])

    ax[0].legend(loc = 2)
    ax[0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')

    ax[0].set_ylim(10**(-28), 2*10**(-27))
    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0]
    paw_0_mask = (pad_params_energy[:,2] > 0)
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)


    if r_beam_params != None :
        ax[1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
        ax[1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    # print(swa_energy[paw_180 < 0])
    # print('1', np.argmin(np.abs(swa_energy[paw_180 < 0][0] - swa_energy)))
    ax[1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    ax[1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')


    ax[1].set_ylabel('Pitch angle width [deg.]')
    ax[1].set_xlabel('Energy [eV]')

    ax[1].set_xlim(20, 1000)
    ax[1].set_xscale('log')
    ax[1].set_ylim(0, 100)
    ax[1].legend()

    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    norm = mpl.colors.BoundaryNorm(np.flipud(swa_energy), color_lines.N)
    sm = plt.cm.ScalarMappable(cmap = color_lines, norm = norm)
    color_lines = color_lines(np.linspace(0, 1, len(swa_energy) + 1))

    for jj in range(np.shape(pad_fit_energy)[0]) :
        ax[2].plot(pitch_angles, np.array(pad_fit_energy)[jj,:], color = color_lines[ - jj])
        ax[2].scatter(pitch_angle_centers, pad[jj,:,0], color = color_lines[ - jj], s = s_set)

    ax[2].set_ylim(10**(-31), 10**(-23))

    ax[2].set_yscale('log')

    cbar = fig.colorbar(sm, ax=ax[2], aspect = 35)#, ticks = [0.01,0.1,1.0,10])
    cbar.set_label(r'Pitch angle: $\alpha$ [deg.]', rotation = 270, labelpad = 15)

    plt.savefig(plot_path+"fit_PA_by_energy"+str(i)+".pdf", format='pdf', bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

    return

def plot_vdf_nobeam_final() :

    ''' redefine the individual VDFs with the final parameters '''
    core_final = lmfit.Parameters()
    core_final.add('n', r_vdf.params['n_c'].value)
    core_final.add('u_par', r_vdf.params['u_par_c'].value)
    core_final.add('v_th_par', r_vdf.params['v_th_par_c'].value)
    core_final.add('v_th_perp', r_vdf.params['v_th_perp_c'].value)
    final_fit_core = core.eval(core_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    beam_final = lmfit.Parameters()
    beam_final.add('n', 0)
    beam_final.add('u_par', 0)
    beam_final.add('v_th_par', 0)
    beam_final.add('v_th_perp', 0)

    halo_final = lmfit.Parameters()
    halo_final.add('n', r_vdf.params['n_h'].value)
    halo_final.add('u_par', r_vdf.params['u_par_h'].value)
    halo_final.add('v_th_par', r_vdf.params['v_th_par_h'].value)
    halo_final.add('v_th_perp', r_vdf.params['v_th_perp_h'].value)
    halo_final.add('kappa', r_vdf.params['kappa'].value)
    final_fit_halo = halo_hallow.eval(halo_final, v_par = v_par_mesh, v_perp = v_perp_mesh)
    # final_fit_halo = halo.eval(halo_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    ''' plot results '''
    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    color_lines = color_lines(np.linspace(0, 1, 4 + 1))

    fig, ax = plt.subplots(2, 2, figsize = (8,8), constrained_layout=True)
    # fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

    ''' full data (not selected by halo, strahl, core energy ranges) '''
    j = 1
    temp = pad[:,-j,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.8 :
        j += 1
        temp = pad[:,-j-1,0]
    jj = 1
    temp = pad[:,jj,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.8 :
        jj += 1
        temp = pad[:,jj,0]
    v_par_pa_idx, anti_v_par_pa_idx, v_perp_pa_idx = jj, -1*j, np.shape(pad_in)[2]//2
    ax[0,0].scatter(swa_energy, pad[:,v_par_pa_idx,0], s = s_set, color = '0.8')
    ax[0,0].scatter(-1*swa_energy, pad[:,anti_v_par_pa_idx,0], s = s_set, color = '0.8')
    ax[0,1].scatter(swa_energy, pad[:,v_perp_pa_idx,0], s = s_set, color = '0.8')

    ''' PARALLEL '''
    ''' SC electron '''
    ax[0,0].scatter(swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, v_par_pa_idx, 0], s = s_set, color = 'k')
    ax[0,0].scatter(-1*swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, anti_v_par_pa_idx,0], s = s_set, color = 'k')

    ''' core v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' halo v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx,0], s = s_set, color = color_lines[1])
    for j in range(1,10) :
        ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx+j,0], s = s_set, color = color_lines[1])

    ''' core v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(-1*swa_energy[high_c:low_c], pad[high_c:low_c,anti_v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' halo v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx,0], s = s_set, color = color_lines[1])
    for j in range(1,10) :
        ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx-j,0], s = s_set, color = color_lines[1])

    ''' full model vdf_1beam, beam set to zero'''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    ''' noise '''
    ax[0,0].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)
    ax[0,0].plot(-1*swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)

    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)

    ax[0,0].set_xlabel('Energy [eV]')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    ax[0,0].set_xlim(-5000, 5000)

    ax[0,0].legend(loc = 8)

    ticks = ax[0,0].get_xticks()
    ticks = np.delete(ticks, (3,5))
    ax[0,0].set_xticks(ticks)
    ax[0,0].annotate('Parallel direction', xy = (0.63,0.94), xycoords = 'axes fraction')
    if B_avg[i,0] < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
    if B_avg[i,0] > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.88), xycoords = 'axes fraction')

    ax[0,0].minorticks_on()

    ''' PERP '''
    ''' SC electron '''
    ax[0,1].scatter(swa_energy[high_sc:low_sc], pad[high_sc:low_sc,v_perp_pa_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    ''' core v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), final_fit_core[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[0], label = 'Core fit')
    ax[0,1].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_perp_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')

    ''' halo v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), final_fit_halo[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2,0], s = s_set, color = color_lines[1], label = 'Halo')
    for j in range(1,10) :
        ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2+j,0], s = s_set, color = color_lines[1])

    ''' full vdf_1beam '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), fit_vdf[:,np.shape(fit_core)[1]//2], lw = lw_set*1.5, ls = ':', color = color_lines[3], label = 'Model')

    ax[0,1].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')

    ax[0,1].set_xlabel('Energy [eV]')
    ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')

    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    ax[0,1].set_xlim(0.8, 5000)

    ax[0,1].annotate('Perpendicular direction', xy = (0.07,0.04), xycoords = 'axes fraction')


    ''' Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', ms = s_set, elinewidth = 0.7, capsize = 1, color = color_lines[0], label = 'VDF averaged ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[1,0].set_xlabel('Pitch angle [deg.]')

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0]
    paw_0_mask = (pad_params_energy[:,2] > 0)
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)


    if r_beam_params != None :
        ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
        ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    # print(swa_energy[paw_180 < 0])
    # print('1', np.argmin(np.abs(swa_energy[paw_180 < 0][0] - swa_energy)))
    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')

    ax[1,1].set_ylabel('Pitch angle width [deg.]')
    ax[1,1].set_xlabel('Energy [eV]')

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    plt.savefig(plot_path+"fit_demo_final_"+str(i)+"_.pdf", format='pdf', bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

def plot_vdf_1beam_final() :

    ''' redefine the individual VDFs with the final parameters '''
    core_final = lmfit.Parameters()
    core_final.add('n', r_vdf.params['n_c'].value)
    core_final.add('u_par', r_vdf.params['u_par_c'].value)
    core_final.add('v_th_par', r_vdf.params['v_th_par_c'].value)
    core_final.add('v_th_perp', r_vdf.params['v_th_perp_c'].value)
    final_fit_core = core.eval(core_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    beam_final = lmfit.Parameters()
    beam_final.add('n', r_vdf.params['n_b'].value)
    beam_final.add('u_par', r_vdf.params['u_par_b'].value)
    beam_final.add('v_th_par', r_vdf.params['v_th_par_b'].value)
    beam_final.add('v_th_perp', r_vdf.params['v_th_perp_b'].value)
    beam_final.add('kappa', r_vdf.params['kappa_b'].value)
    final_fit_beam = beam.eval(beam_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    halo_final = lmfit.Parameters()
    halo_final.add('n', r_vdf.params['n_h'].value)
    halo_final.add('u_par', r_vdf.params['u_par_h'].value)
    halo_final.add('v_th_par', r_vdf.params['v_th_par_h'].value)
    halo_final.add('v_th_perp', r_vdf.params['v_th_perp_h'].value)
    halo_final.add('kappa', r_vdf.params['kappa'].value)
    final_fit_halo = halo_hallow.eval(halo_final, v_par = v_par_mesh, v_perp = v_perp_mesh)
    # final_fit_halo = halo.eval(halo_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    ''' plot results '''
    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    color_lines = color_lines(np.linspace(0, 1, 4 + 1))

    fig, ax = plt.subplots(2, 2, figsize = (8,8), constrained_layout=True)
    # fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

    ''' full data (not selected by halo, strahl, core energy ranges) '''
    j = 1
    temp = pad[:,-j,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.9 :
        j += 1
        temp = pad[:,-j-1,0]
    jj = 0
    temp = pad[:,jj,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.9 :
        jj += 1
        temp = pad[:,jj,0]
    v_par_pa_idx, anti_v_par_pa_idx, v_perp_pa_idx = jj, -1*j, np.shape(pad_in)[2]//2
    ax[0,0].scatter(swa_energy, pad[:,0,0], s = s_set, color = '0.8')
    ax[0,0].scatter(-1*swa_energy, pad[:,anti_v_par_pa_idx,0], s = s_set, color = '0.8')
    ax[0,1].scatter(swa_energy, pad[:,v_perp_pa_idx,0], s = s_set, color = '0.8')

    ''' PARALLEL '''
    ''' SC electron '''
    ax[0,0].scatter(swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, v_par_pa_idx, 0], s = s_set, color = 'k')
    ax[0,0].scatter(-1*swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, anti_v_par_pa_idx,0], s = s_set, color = 'k')

    ''' core v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' halo v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx,0], s = s_set, color = color_lines[1])
    for j in range(1,10) :
        ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx+j,0], s = s_set, color = color_lines[1])

    ''' core v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(-1*swa_energy[high_c:low_c], pad[high_c:low_c,anti_v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' halo v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx,0], s = s_set, color = color_lines[1])
    for j in range(4,10) :
        ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx-j,0], s = s_set, color = color_lines[1])

    ''' anti par beam '''
    if anti_par_strahl_cond == True :
        ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_beam[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Anti-par beam fit')
        for j in range(np.abs(anti_v_par_pa_idx)+1) :
            ax[0,0].scatter(-1*swa_energy[high_b:low_b], pad[high_b:low_b,-j,0], s = s_set, color = color_lines[2], label = 'Anti-par beam' if j == 0 else '')
    ''' beam '''
    if par_strahl_cond == True :
        ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_beam[-1,v_par_arr>0], lw = lw_set, color = color_lines[3], label = 'Par beam fit')
        for j in range(np.abs(v_par_pa_idx)+1) :
            ax[0,0].scatter(swa_energy[high_b:low_b], pad[high_b:j,0], s = s_set, color = color_lines[3], label = 'Par beam' if j == 0 else '')

    ''' full model vdf_1beam '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    ''' noise '''
    ax[0,0].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)
    ax[0,0].plot(-1*swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)

    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)

    ax[0,0].set_xlabel('Energy [eV]')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    ax[0,0].set_xlim(-5000, 5000)

    ax[0,0].legend(loc = 8)

    ticks = ax[0,0].get_xticks()
    ticks = np.delete(ticks, (3,5))
    ax[0,0].set_xticks(ticks)

    ax[0,0].annotate('Parallel direction', xy = (0.64,0.94), xycoords = 'axes fraction')
    if B_avg[i,0] < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
    if B_avg[i,0] > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.88), xycoords = 'axes fraction')


    ax[0,0].minorticks_on()

    ''' PERP '''
    ''' SC electron '''
    ax[0,1].scatter(swa_energy[high_sc:low_sc], pad[high_sc:low_sc,v_perp_pa_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    ''' core v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), final_fit_core[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[0], label = 'Core fit')
    ax[0,1].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_perp_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')

    ''' halo v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), final_fit_halo[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2,0], s = s_set, color = color_lines[1], label = 'Halo')
    for j in range(1,10) :
        ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2+j,0], s = s_set, color = color_lines[1])

    ''' full vdf_1beam '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), fit_vdf[:,np.shape(fit_core)[1]//2], lw = lw_set*1.5, ls = ':', color = color_lines[3], label = 'Model')

    ax[0,1].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')

    ax[0,1].set_xlabel('Energy [eV]')
    ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')

    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    ax[0,1].set_xlim(0.8, 5000)

    ax[0,1].annotate('Perpendicular direction', xy = (0.07,0.04), xycoords = 'axes fraction')


    ''' Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', ms = s_set, elinewidth = 0.7, capsize = 1, color = color_lines[0], label = 'VDF averaged ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[1,0].set_xlabel('Pitch angle [deg.]')

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0]
    paw_0_mask = (pad_params_energy[:,2] > 0)
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)


    if r_beam_params != None :
        ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
        ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    # print(swa_energy[paw_180 < 0])
    # print('1', np.argmin(np.abs(swa_energy[paw_180 < 0][0] - swa_energy)))
    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')

    ax[1,1].set_ylabel('Pitch angle width [deg.]')
    ax[1,1].set_xlabel('Energy [eV]')

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    plt.savefig(plot_path+"fit_demo_final_"+str(i)+"_.pdf", format='pdf', bbox_inches = 'tight')
    # pickle.dump(fig, open(plot_path+'myplot_fig.pickle', 'wb'))
    plt.close(plt.gcf())
    fig.clf()

def plot_vdf_2beam_final() :

    ''' redefine the individual VDFs with the final parameters '''
    core_final = lmfit.Parameters()
    core_final.add('n', r_vdf.params['n_c'].value)
    core_final.add('u_par', r_vdf.params['u_par_c'].value)
    core_final.add('v_th_par', r_vdf.params['v_th_par_c'].value)
    core_final.add('v_th_perp', r_vdf.params['v_th_perp_c'].value)
    final_fit_core = core.eval(core_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    beam_final_par = lmfit.Parameters()
    beam_final_par.add('n', r_vdf.params['n_b_par'].value)
    beam_final_par.add('u_par', r_vdf.params['u_par_b_par'].value)
    beam_final_par.add('v_th_par', r_vdf.params['v_th_par_b_par'].value)
    beam_final_par.add('v_th_perp', r_vdf.params['v_th_perp_b_par'].value)
    beam_final_par.add('kappa', r_vdf.params['kappa_b_par'].value)
    final_fit_beam_par = beam.eval(beam_final_par, v_par = v_par_mesh, v_perp = v_perp_mesh)

    beam_final_anti_par = lmfit.Parameters()
    beam_final_anti_par.add('n', r_vdf.params['n_b_anti_par'].value)
    beam_final_anti_par.add('u_par', r_vdf.params['u_par_b_anti_par'].value)
    beam_final_anti_par.add('v_th_par', r_vdf.params['v_th_par_b_anti_par'].value)
    beam_final_anti_par.add('v_th_perp', r_vdf.params['v_th_perp_b_anti_par'].value)
    beam_final_anti_par.add('kappa', r_vdf.params['kappa_b_anti_par'].value)
    final_fit_beam_anti_par = beam.eval(beam_final_anti_par, v_par = v_par_mesh, v_perp = v_perp_mesh)

    halo_final = lmfit.Parameters()
    halo_final.add('n', r_vdf.params['n_h'].value)
    halo_final.add('u_par', r_vdf.params['u_par_h'].value)
    halo_final.add('v_th_par', r_vdf.params['v_th_par_h'].value)
    halo_final.add('v_th_perp', r_vdf.params['v_th_perp_h'].value)
    halo_final.add('kappa', r_vdf.params['kappa'].value)
    final_fit_halo = halo_hallow.eval(halo_final, v_par = v_par_mesh, v_perp = v_perp_mesh)
    # final_fit_halo = halo.eval(halo_final, v_par = v_par_mesh, v_perp = v_perp_mesh)

    ''' plot results '''
    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    color_lines = color_lines(np.linspace(0, 1, 4 + 1))

    fig, ax = plt.subplots(2, 2, figsize = (8,8), constrained_layout=True)
    # fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

    ''' full data (not selected by halo, strahl, core energy ranges) '''
    j = 1
    temp = pad[:,-j,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.9 :
        j += 1
        temp = pad[:,-j-1,0]
    jj = 0
    temp = pad[:,jj,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.9 :
        jj += 1
        temp = pad[:,jj,0]
    v_par_pa_idx, anti_v_par_pa_idx, v_perp_pa_idx = jj, -1*j, np.shape(pad_in)[2]//2
    ax[0,0].scatter(swa_energy, pad[:,0,0], s = s_set, color = '0.8')
    ax[0,0].scatter(-1*swa_energy, pad[:,anti_v_par_pa_idx,0], s = s_set, color = '0.8')
    ax[0,1].scatter(swa_energy, pad[:,v_perp_pa_idx,0], s = s_set, color = '0.8')

    ''' PARALLEL '''
    ''' SC electron '''
    ax[0,0].scatter(swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, v_par_pa_idx, 0], s = s_set, color = 'k')
    ax[0,0].scatter(-1*swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, anti_v_par_pa_idx,0], s = s_set, color = 'k')

    ''' core v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' halo v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx,0], s = s_set, color = color_lines[1])
    for j in range(1,10) :
        ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx+j,0], s = s_set, color = color_lines[1])

    ''' core v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(-1*swa_energy[high_c:low_c], pad[high_c:low_c,anti_v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' halo v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx,0], s = s_set, color = color_lines[1])
    for j in range(1,10) :
        ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx-j,0], s = s_set, color = color_lines[1])

    ''' beam '''
    ''' anti par beam '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_beam_anti_par[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Anti-par beam fit')
    for j in range(np.abs(anti_v_par_pa_idx)+1) :
        ax[0,0].scatter(-1*swa_energy[high_b:low_b], pad[high_b:low_b,-j,0], s = s_set, color = color_lines[2], label = 'Anti-par beam' if j == 0 else '')
    ''' beam '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_beam_par[-1,v_par_arr>0], lw = lw_set, color = color_lines[3], label = 'Par beam fit')
    for j in range(np.abs(v_par_pa_idx)+1) :
        ax[0,0].scatter(swa_energy[high_b:low_b], pad[high_b:low_b,j,0], s = s_set, color = color_lines[3], label = 'Par beam' if j == 0 else '')

    ''' full model vdf_1beam '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    ''' noise '''
    ax[0,0].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)
    ax[0,0].plot(-1*swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)

    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)

    ax[0,0].set_xlabel('Energy [eV]')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    ax[0,0].set_xlim(-5000, 5000)

    ax[0,0].legend(loc = 8)

    ticks = ax[0,0].get_xticks()
    ticks = np.delete(ticks, (3,5))
    ax[0,0].set_xticks(ticks)
    ax[0,0].annotate('Parallel direction', xy = (0.63,0.94), xycoords = 'axes fraction')
    if B_avg[i,0] < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
    if B_avg[i,0] > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.88), xycoords = 'axes fraction')

    ax[0,0].minorticks_on()

    ''' PERP '''
    ''' SC electron '''
    ax[0,1].scatter(swa_energy[high_sc:low_sc], pad[high_sc:low_sc,v_perp_pa_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    ''' core v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), final_fit_core[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[0], label = 'Core fit')
    ax[0,1].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_perp_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')

    ''' halo v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), final_fit_halo[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2,0], s = s_set, color = color_lines[1], label = 'Halo')
    for j in range(1,10) :
        ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2+j,0], s = s_set, color = color_lines[1])

    ''' full vdf_1beam '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), fit_vdf[:,np.shape(fit_core)[1]//2], lw = lw_set*1.5, ls = ':', color = color_lines[3], label = 'Model')

    ax[0,1].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')

    ax[0,1].set_xlabel('Energy [eV]')
    ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')

    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    ax[0,1].set_xlim(0.8, 5000)

    ax[0,1].annotate('Perpendicular direction', xy = (0.07,0.04), xycoords = 'axes fraction')


    ''' Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', ms = s_set, elinewidth = 0.7, capsize = 1, color = color_lines[0], label = 'VDF averaged ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[1,0].set_xlabel('Pitch angle [deg.]')

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0]
    paw_0_mask = (pad_params_energy[:,2] > 0)
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)


    if r_beam_params != None :
        ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
        ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    # print(swa_energy[paw_180 < 0])
    # print('1', np.argmin(np.abs(swa_energy[paw_180 < 0][0] - swa_energy)))
    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')


    ax[1,1].set_ylabel('Pitch angle width [deg.]')
    ax[1,1].set_xlabel('Energy [eV]')

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    plt.savefig(plot_path+"fit_demo_final_"+str(i)+"_.pdf", format='pdf', bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

def core_fit_function(psd_data, v_par, v_perp, core_const, core_init) :

    ''' set initial core guess '''
    p_core = lmfit.Parameters()
    p_core.add('n', core_init[0], vary = False, min = core_const[0][0], max = core_const[0][1])
    p_core.add('u_par', core_init[1], vary = False, min = core_const[1][0], max = core_const[1][1])
    p_core.add('v_th_par', core_init[2], vary = False, min = core_const[2][0], max = core_const[2][1])
    p_core.add('v_th_perp', core_init[3], vary = False, min = core_const[2][0], max = core_const[3][1])

    p_list = list(p_core.valuesdict().keys())


    try :
        r_core = core.fit(psd_data, p_core, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
    except ValueError :
        print('Core dint fit')
        r_core = None

    def fit_update(p_core, r_core, vary_list, psd_data, v_par, v_perp, core_const) :

        p_core.add('n', r_core.params['n'].value, vary = vary_list[0], min = core_const[0][0], max = core_const[0][1])
        p_core.add('u_par', r_core.params['u_par'].value, vary = vary_list[1], min = core_const[1][0], max = core_const[1][1])
        p_core.add('v_th_par', r_core.params['v_th_par'].value, vary = vary_list[2], min = core_const[2][0], max = core_const[2][1])
        p_core.add('v_th_perp', r_core.params['v_th_perp'].value, vary = vary_list[3], min = core_const[3][0], max = core_const[3][1])

        try :
            r_core = core.fit(psd_data, p_core, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
        except ValueError :
            print('Core dint fit')
            r_core = None

        return r_core

    vary_list = [[True, True, False, False], \
    [True, False, True, False], \
    [True, False, False, True], \
    # [False, True, False, True], \
    # [False, False, False, True]]#, \
    [True, True, True, True]]

    for i in range(len(vary_list)) :
        if r_core != None :
            r_core = fit_update(p_core, r_core, vary_list[i], psd_data, v_par, v_perp, core_const)

    if r_core != None :
        fit_core = core.eval(r_core.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
        ''' if no errors were generated or if stderr is zero '''
        if r_core.errorbars == False :
            for i in range(len(p_list)) :
                r_core.params[p_list[i]].stderr = stderr_fact*np.abs(r_core.params[p_list[i]].value)

        for i in range(len(p_list)) :
            if r_core.params[p_list[i]].stderr == 0 :
                r_core.params[p_list[i]].stderr = stderr_fact*np.abs(r_core.params[p_list[i]].value)
    else :
        fit_core = core.eval(p_core, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_core, fit_core

def beam_fit_function(psd_data, v_par, v_perp, counts, beam_const, beam_init) :

    ''' set initial beam guess '''
    p_beam = lmfit.Parameters()
    p_beam.add('n', beam_init[0], vary = False, min = beam_const[0][0], max = beam_const[0][1])
    p_beam.add('u_par', beam_init[1], vary = False, min = beam_const[1][0], max = beam_const[1][1])
    p_beam.add('v_th_par', beam_init[2], vary = False, min = beam_const[2][0], max = beam_const[2][1])
    p_beam.add('v_th_perp', beam_init[3], vary = False, min = beam_const[3][0], max = beam_const[3][1])
    p_beam.add('kappa', beam_init[4], vary = False, min = beam_const[4][0], max = beam_const[4][1])

    p_list = list(p_beam.valuesdict().keys())

    try :
        r_beam = beam.fit(psd_data, p_beam, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300, weights = 1)
    except ValueError :
        # print('beam dint fit')
        r_beam = None

    def fit_update(p_beam, r_beam, vary_list, psd_data, v_par, v_perp, beam_const) :

        p_beam.add('n', r_beam.params['n'].value, vary = vary_list[0], min = beam_const[0][0], max = beam_const[0][1])
        p_beam.add('u_par', r_beam.params['u_par'].value, vary = vary_list[1], min = beam_const[1][0], max = beam_const[1][1])
        p_beam.add('v_th_par', r_beam.params['v_th_par'].value, vary = vary_list[2], min = beam_const[2][0], max = beam_const[2][1])
        p_beam.add('v_th_perp', r_beam.params['v_th_perp'].value, vary = vary_list[3], min = beam_const[3][0], max = beam_const[3][1])
        p_beam.add('kappa', r_beam.params['kappa'].value, vary = vary_list[4], min = beam_const[4][0], max = beam_const[4][1])

        weight_set = 1/np.sqrt(counts)
        weight_set[np.isinf(weight_set)] = 0.0
        # weight_set = 1

        try :
            r_beam = beam.fit(psd_data, p_beam, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300, weights = weight_set)
        except ValueError :
            # print('beam dint fit')
            r_beam = None

        return r_beam

    vary_list = [[True, True, False, False, False], \
                [True, False, False, True, False], \
                # [False, False, True, True, False], \
                [True, False, True, False, False], \
                [True, False, True, True, True]]

    # for i in range(len(vary_list)) :
    for i in range(3) :
        if r_beam != None :
            r_beam = fit_update(p_beam, r_beam, vary_list[i], psd_data, v_par, v_perp, beam_const)

    if r_beam != None :
        fit_beam = beam.eval(r_beam.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
        if r_beam.errorbars == False :
            for i in range(len(p_list)) :
                r_beam.params[p_list[i]].stderr = stderr_fact*np.abs(r_beam.params[p_list[i]].value)

        for i in range(len(p_list)) :
            if r_beam.params[p_list[i]].stderr == 0 :
                r_beam.params[p_list[i]].stderr = stderr_fact*np.abs(r_beam.params[p_list[i]].value)
    else :
        fit_beam = beam.eval(p_beam, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_beam, fit_beam

def halo_fit_function(psd_data, v_par, v_perp, halo_const, halo_init) :

    ''' set initial halo guess '''
    p_halo = lmfit.Parameters()
    p_halo.add('n', halo_init[0], vary = True, min = halo_const[0][0], max = halo_const[0][1])
    p_halo.add('u_par', halo_init[1], vary = False, min = halo_const[1][0], max = halo_const[1][1])
    p_halo.add('v_th_par', halo_init[2], vary = False, min = halo_const[2][0], max = halo_const[2][1])
    p_halo.add('v_th_perp', halo_init[3], vary = False, min = halo_const[3][0], max = halo_const[3][1])
    p_halo.add('kappa', halo_init[4], vary = False, min = halo_const[4][0], max = halo_const[4][1])

    p_list = list(p_halo.valuesdict().keys())

    try :
        r_halo = halo.fit(psd_data, p_halo, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
    except ValueError :
        print('1 Halo dint fit')
        r_halo = None

    def fit_update(jj, p_halo, r_halo, vary_list, psd_data, v_par, v_perp, halo_const) :

        p_halo.add('n', r_halo.params['n'].value, vary = vary_list[0], min = halo_const[0][0], max = halo_const[0][1])
        p_halo.add('u_par', r_halo.params['u_par'].value, vary = vary_list[1], min = halo_const[1][0], max = halo_const[1][1])
        p_halo.add('v_th_par', r_halo.params['v_th_par'].value, vary = vary_list[2], min = halo_const[2][0], max = halo_const[2][1])
        p_halo.add('v_th_perp', r_halo.params['v_th_perp'].value, vary = vary_list[3], min = halo_const[3][0], max = halo_const[3][1])
        p_halo.add('kappa', r_halo.params['kappa'].value, vary = vary_list[4], min = halo_const[4][0], max = halo_const[4][1])

        try :
            r_halo = halo.fit(psd_data, p_halo, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
        except ValueError :
            print(jj, 'Halo dint fit')
            r_halo = None

        return r_halo

    vary_list = [[True, True, False, False, False], \
        [True, False, True, True, False], \
        [False, False, True, True, True], \
        [False, False, False, True, False]]

    r_halo_save = []
    for ii in range(len(vary_list)) :
        if r_halo != None :
            r_halo = fit_update(ii+2, p_halo, r_halo, vary_list[ii], psd_data, v_par, v_perp, halo_const)
            r_halo_save.append(r_halo)
        else :
            print('Final halo fit doesnt work')
            r_halo = r_halo_save[ii-2]
            break

    if r_halo != None :
        fit_halo = halo.eval(r_halo.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
        if r_halo.errorbars == False :
            for i in range(len(p_list)) :
                r_halo.params[p_list[i]].stderr = stderr_fact*np.abs(r_halo.params[p_list[i]].value)

        for i in range(len(p_list)) :
            if r_halo.params[p_list[i]].stderr == 0 :
                r_halo.params[p_list[i]].stderr = stderr_fact*np.abs(r_halo.params[p_list[i]].value)
    else:
        fit_halo = halo.eval(p_halo, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_halo, fit_halo

def vdf_fit_function_one_beam(psd_data, v_par, v_perp, n, r_core, r_beam_params, r_halo) :

    vdf_const = []

    if anti_par_strahl_cond == True :
        r_beam = r_beam_anti_par
    if par_strahl_cond == True :
        r_beam = r_beam_par

    ''' use standard errors as bounds for these varibales '''
    vdf_const.append([-1*r_core.params['n'].stderr + r_core.params['n'].value, r_core.params['n'].stderr + r_core.params['n'].value])
    vdf_const.append([-1*r_core.params['u_par'].stderr + r_core.params['u_par'].value, r_core.params['u_par'].stderr + r_core.params['u_par'].value])
    vdf_const.append([-1*r_core.params['v_th_par'].stderr + r_core.params['v_th_par'].value, r_core.params['v_th_par'].stderr + r_core.params['v_th_par'].value])
    vdf_const.append([-1*r_core.params['v_th_perp'].stderr + r_core.params['v_th_perp'].value, r_core.params['v_th_perp'].stderr + r_core.params['v_th_perp'].value])
    if r_beam_anti_par != None :
        vdf_const.append([-1*r_beam.params['n'].stderr + r_beam.params['n'].value, r_beam.params['n'].stderr + r_beam.params['n'].value])
        vdf_const.append([-1*r_beam.params['u_par'].stderr + r_beam.params['u_par'].value, r_beam.params['u_par'].stderr + r_beam.params['u_par'].value])
        vdf_const.append([-1*r_beam.params['v_th_par'].stderr + r_beam.params['v_th_par'].value, r_beam.params['v_th_par'].stderr + r_beam.params['v_th_par'].value])
        vdf_const.append([-1*r_beam.params['v_th_perp'].stderr + r_beam.params['v_th_perp'].value, r_beam.params['v_th_perp'].stderr + r_beam.params['v_th_perp'].value])
        vdf_const.append([-1*r_beam.params['kappa'].stderr + r_beam.params['kappa'].value, r_beam.params['kappa'].stderr + r_beam.params['kappa'].value])
    else :
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
    vdf_const.append([-1*r_halo.params['n'].stderr + r_halo.params['n'].value, r_halo.params['n'].stderr + r_halo.params['n'].value])
    vdf_const.append([-1*r_halo.params['u_par'].stderr + r_halo.params['u_par'].value, r_halo.params['u_par'].stderr + r_halo.params['u_par'].value])
    vdf_const.append([-1*r_halo.params['v_th_par'].stderr + r_halo.params['v_th_par'].value, r_halo.params['v_th_par'].stderr + r_halo.params['v_th_par'].value])
    vdf_const.append([-1*r_halo.params['v_th_perp'].stderr + r_halo.params['v_th_perp'].value, r_halo.params['v_th_perp'].stderr + r_halo.params['v_th_perp'].value])
    vdf_const.append([-1*r_halo.params['kappa'].stderr + r_halo.params['kappa'].value, r_halo.params['kappa'].stderr + r_halo.params['kappa'].value])

    for j in range(2) : vdf_const.append([None, None])

    p_vdf = lmfit.Parameters()
    p_vdf.add('n_c', r_core.params['n'], vary = False, min = vdf_const[0][0], max = vdf_const[0][1])
    p_vdf.add('u_par_c', r_core.params['u_par'], vary = False, min = vdf_const[1][0], max = vdf_const[1][1])
    p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary = False, min = vdf_const[2][0], max = vdf_const[2][1])
    p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary = False, min = vdf_const[3][0], max = vdf_const[3][1])
    if r_beam != None :
        p_vdf.add('n_b', r_beam.params['n'], vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
        p_vdf.add('u_par_b', r_beam.params['u_par'], vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
        p_vdf.add('v_th_par_b', r_beam.params['v_th_par'], vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
        p_vdf.add('v_th_perp_b', r_beam.params['v_th_perp'], vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
        p_vdf.add('kappa_b', r_beam.params['kappa'], vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
    else :
        p_vdf.add('n_b', 0.0, vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
        p_vdf.add('u_par_b', 0.0, vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
        p_vdf.add('v_th_par_b', 1.0, vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
        p_vdf.add('v_th_perp_b', 1.0, vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
        p_vdf.add('kappa_b', 3, vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
    p_vdf.add('n_h', r_halo.params['n'], vary = False, min = vdf_const[9][0], max = vdf_const[9][1])
    p_vdf.add('u_par_h', r_halo.params['u_par'], vary = False, min = vdf_const[10][0], max = vdf_const[10][1], expr = '(u_par_c*n_c + u_par_b*n_b)/n_h')
    p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary = False, min = vdf_const[11][0], max = vdf_const[11][1])
    p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary = False, min = vdf_const[12][0], max = vdf_const[12][1])
    p_vdf.add('kappa', r_halo.params['kappa'], vary = False, min = vdf_const[13][0], max = vdf_const[13][1])
    p_vdf.add('n_sc', n, vary = False, min = vdf_const[14][0], max = vdf_const[14][1])

    try :
        r_vdf = vdf_1beam.fit(psd_data, p_vdf, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
    except (ValueError, TypeError) :
        r_vdf = None

    def fit_update(p_vdf, r_vdf, vary_list, psd_data, v_par, v_perp, vdf_const) :

        p_vdf.add('n_c', r_core.params['n'], vary = vary_list[0], min = vdf_const[0][0], max = vdf_const[0][1])
        p_vdf.add('u_par_c', r_core.params['u_par'], vary = vary_list[1], min = vdf_const[1][0], max = vdf_const[1][1])
        p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary = vary_list[2], min = vdf_const[2][0], max = vdf_const[2][1])
        p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary = vary_list[3], min = vdf_const[3][0], max = vdf_const[3][1])
        if r_beam != None :
            p_vdf.add('n_b', r_beam.params['n'], vary = vary_list[4], min = vdf_const[4][0], max = vdf_const[4][1])
            p_vdf.add('u_par_b', r_beam.params['u_par'], vary = vary_list[5], min = vdf_const[5][0], max = vdf_const[5][1])
            p_vdf.add('v_th_par_b', r_beam.params['v_th_par'], vary = vary_list[6], min = vdf_const[6][0], max = vdf_const[6][1])
            p_vdf.add('v_th_perp_b', r_beam.params['v_th_perp'], vary = vary_list[7], min = vdf_const[7][0], max = vdf_const[7][1])
            p_vdf.add('kappa_b', r_beam.params['kappa'], vary = vary_list[8], min = vdf_const[8][0], max = vdf_const[8][1])
        else :
            p_vdf.add('n_b', 0.0, vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
            p_vdf.add('u_par_b', 0.0, vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
            p_vdf.add('v_th_par_b', 1.0, vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
            p_vdf.add('v_th_perp_b', 1.0, vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
            p_vdf.add('kappa_b', 3, vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
        p_vdf.add('n_h', r_halo.params['n'], vary = vary_list[9], min = vdf_const[9][0], max = vdf_const[9][1])
        p_vdf.add('u_par_h', r_halo.params['u_par'], vary = vary_list[10], min = vdf_const[10][0], max = vdf_const[10][1], expr = '(u_par_c*n_c + u_par_b*n_b)/n_h')
        p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary = vary_list[11], min = vdf_const[11][0], max = vdf_const[11][1])
        p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary = vary_list[12], min = vdf_const[12][0], max = vdf_const[12][1])
        p_vdf.add('kappa', r_halo.params['kappa'], vary = vary_list[13], min = vdf_const[13][0], max = vdf_const[13][1])
        p_vdf.add('n_sc', n, vary = False, min = vdf_const[14][0], max = vdf_const[14][1])

        try :
            r_vdf = vdf_1beam.fit(psd_data, p_vdf, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
        except ValueError :
            # print('2 vdf_1beam dint fit')
            r_vdf = None

        return r_vdf

    if r_beam != None :
        vary_list = [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False]]
    else :
        vary_list = [[True, True, True, True, False, False, False, False, False, True, True, True, True, True, False]]

    for ii in range(len(vary_list)) :
        if r_vdf != None :
            r_vdf = fit_update(p_vdf, r_vdf, vary_list[ii], psd_data, v_par, v_perp, vdf_const)
        else :
            print('final fit is not working')

    if r_vdf != None :
        fit_vdf = vdf_1beam.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    else:
        fit_vdf = vdf_1beam.eval(p_vdf, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_vdf, fit_vdf

def vdf_fit_function_two_beam(psd_data, v_par, v_perp, n, r_core, r_beam_params, r_halo) :

    ''' set initial halo guess '''

    vdf_const = []

    ''' use standard errors as bounds for these varibales '''
    vdf_const.append([-1*r_core.params['n'].stderr + r_core.params['n'].value, r_core.params['n'].stderr + r_core.params['n'].value])
    vdf_const.append([-1*r_core.params['u_par'].stderr + r_core.params['u_par'].value, r_core.params['u_par'].stderr + r_core.params['u_par'].value])
    vdf_const.append([-1*r_core.params['v_th_par'].stderr + r_core.params['v_th_par'].value, r_core.params['v_th_par'].stderr + r_core.params['v_th_par'].value])
    vdf_const.append([-1*r_core.params['v_th_perp'].stderr + r_core.params['v_th_perp'].value, r_core.params['v_th_perp'].stderr + r_core.params['v_th_perp'].value])
    if r_beam_par != None :
        vdf_const.append([-1*r_beam_par.params['n'].stderr + r_beam_par.params['n'].value, r_beam_par.params['n'].stderr + r_beam_par.params['n'].value])
        vdf_const.append([-1*r_beam_par.params['u_par'].stderr + r_beam_par.params['u_par'].value, r_beam_par.params['u_par'].stderr + r_beam_par.params['u_par'].value])
        vdf_const.append([-1*r_beam_par.params['v_th_par'].stderr + r_beam_par.params['v_th_par'].value, r_beam_par.params['v_th_par'].stderr + r_beam_par.params['v_th_par'].value])
        vdf_const.append([-1*r_beam_par.params['v_th_perp'].stderr + r_beam_par.params['v_th_perp'].value, r_beam_par.params['v_th_perp'].stderr + r_beam_par.params['v_th_perp'].value])
        vdf_const.append([-1*r_beam_par.params['kappa'].stderr + r_beam_par.params['kappa'].value, r_beam_par.params['kappa'].stderr + r_beam_par.params['kappa'].value])
    else :
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
    if r_beam_anti_par != None :
        vdf_const.append([-1*r_beam_anti_par.params['n'].stderr + r_beam_anti_par.params['n'].value, r_beam_anti_par.params['n'].stderr + r_beam_anti_par.params['n'].value])
        vdf_const.append([-1*r_beam_anti_par.params['u_par'].stderr + r_beam_anti_par.params['u_par'].value, r_beam_anti_par.params['u_par'].stderr + r_beam_anti_par.params['u_par'].value])
        vdf_const.append([-1*r_beam_anti_par.params['v_th_par'].stderr + r_beam_anti_par.params['v_th_par'].value, r_beam_anti_par.params['v_th_par'].stderr + r_beam_anti_par.params['v_th_par'].value])
        vdf_const.append([-1*r_beam_anti_par.params['v_th_perp'].stderr + r_beam_anti_par.params['v_th_perp'].value, r_beam_anti_par.params['v_th_perp'].stderr + r_beam_anti_par.params['v_th_perp'].value])
        vdf_const.append([-1*r_beam_anti_par.params['kappa'].stderr + r_beam_anti_par.params['kappa'].value, r_beam_anti_par.params['kappa'].stderr + r_beam_anti_par.params['kappa'].value])
    else :
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
        vdf_const.append([0, 1])
    vdf_const.append([-1*r_halo.params['n'].stderr + r_halo.params['n'].value, r_halo.params['n'].stderr + r_halo.params['n'].value])
    vdf_const.append([-1*r_halo.params['u_par'].stderr + r_halo.params['u_par'].value, r_halo.params['u_par'].stderr + r_halo.params['u_par'].value])
    vdf_const.append([-1*r_halo.params['v_th_par'].stderr + r_halo.params['v_th_par'].value, r_halo.params['v_th_par'].stderr + r_halo.params['v_th_par'].value])
    vdf_const.append([-1*r_halo.params['v_th_perp'].stderr + r_halo.params['v_th_perp'].value, r_halo.params['v_th_perp'].stderr + r_halo.params['v_th_perp'].value])
    vdf_const.append([-1*r_halo.params['kappa'].stderr + r_halo.params['kappa'].value, r_halo.params['kappa'].stderr + r_halo.params['kappa'].value])

    for j in range(2) : vdf_const.append([None, None])

    p_vdf = lmfit.Parameters()
    p_vdf.add('n_c', r_core.params['n'], vary = False, min = vdf_const[0][0], max = vdf_const[0][1])
    p_vdf.add('u_par_c', r_core.params['u_par'], vary = False, min = vdf_const[1][0], max = vdf_const[1][1])
    p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary = False, min = vdf_const[2][0], max = vdf_const[2][1])
    p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary = False, min = vdf_const[3][0], max = vdf_const[3][1])
    if r_beam_par != None :
        p_vdf.add('n_b_par', r_beam_par.params['n'], vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
        p_vdf.add('u_par_b_par', r_beam_par.params['u_par'], vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
        p_vdf.add('v_th_par_b_par', r_beam_par.params['v_th_par'], vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
        p_vdf.add('v_th_perp_b_par', r_beam_par.params['v_th_perp'], vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
        p_vdf.add('kappa_b_par', r_beam_par.params['kappa'], vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
    else :
        p_vdf.add('n_b_par', 0.0, vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
        p_vdf.add('u_par_b_par', 0.0, vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
        p_vdf.add('v_th_par_b_par', 1.0, vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
        p_vdf.add('v_th_perp_b_par', 1.0, vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
        p_vdf.add('kappa_b_par', 10, vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
    if r_beam_anti_par != None :
        p_vdf.add('n_b_anti_par', r_beam_anti_par.params['n'], vary = False, min = vdf_const[9][0], max = vdf_const[9][1])
        p_vdf.add('u_par_b_anti_par', r_beam_anti_par.params['u_par'], vary = False, min = vdf_const[10][0], max = vdf_const[10][1])
        p_vdf.add('v_th_par_b_anti_par', r_beam_anti_par.params['v_th_par'], vary = False, min = vdf_const[11][0], max = vdf_const[11][1])
        p_vdf.add('v_th_perp_b_anti_par', r_beam_anti_par.params['v_th_perp'], vary = False, min = vdf_const[12][0], max = vdf_const[12][1])
        p_vdf.add('kappa_b_anti_par', r_beam_anti_par.params['kappa'], vary = False, min = vdf_const[13][0], max = vdf_const[13][1])
    else :
        p_vdf.add('n_b_anti_par', 0.0, vary = False, min = vdf_const[9][0], max = vdf_const[9][1])
        p_vdf.add('u_par_b_anti_par', 0.0, vary = False, min = vdf_const[10][0], max = vdf_const[10][1])
        p_vdf.add('v_th_par_b_anti_par', 1.0, vary = False, min = vdf_const[11][0], max = vdf_const[11][1])
        p_vdf.add('v_th_perp_b_anti_par', 1.0, vary = False, min = vdf_const[12][0], max = vdf_const[12][1])
        p_vdf.add('kappa_b_anti_par', 10, vary = False, min = vdf_const[13][0], max = vdf_const[13][1])
    p_vdf.add('n_h', r_halo.params['n'], vary = False, min = vdf_const[14][0], max = vdf_const[14][1])
    p_vdf.add('u_par_h', r_halo.params['u_par'], vary = False, min = vdf_const[15][0], max = vdf_const[15][1], expr = '(u_par_c*n_c + u_par_b_par*n_b_par + u_par_b_anti_par*n_b_anti_par)/n_h')
    p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary = False, min = vdf_const[16][0], max = vdf_const[16][1])
    p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary = False, min = vdf_const[17][0], max = vdf_const[17][1])
    p_vdf.add('kappa', r_halo.params['kappa'], vary = False, min = vdf_const[18][0], max = vdf_const[18][1])
    p_vdf.add('n_sc', n, vary = False, min = vdf_const[19][0], max = vdf_const[19][1])

    try :
        r_vdf = vdf_2beam.fit(psd_data, p_vdf, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
    except (ValueError, TypeError) :
        # print('1 vdf_2beam dint fit')
        r_vdf = None

    def fit_update(p_vdf, r_vdf, vary_list, psd_data, v_par, v_perp, vdf_const) :

        p_vdf.add('n_c', r_vdf.params['n_c'], vary = vary_list[0], min = vdf_const[0][0], max = vdf_const[0][1])
        p_vdf.add('u_par_c', r_vdf.params['u_par_c'], vary = vary_list[1], min = vdf_const[1][0], max = vdf_const[1][1])
        p_vdf.add('v_th_par_c', r_vdf.params['v_th_par_c'], vary = vary_list[2], min = vdf_const[2][0], max = vdf_const[2][1])
        p_vdf.add('v_th_perp_c', r_vdf.params['v_th_perp_c'], vary = vary_list[3], min = vdf_const[3][0], max = vdf_const[3][1])
        if r_beam_par != None :
            p_vdf.add('n_b_par', r_vdf.params['n_b_par'], vary = vary_list[4], min = vdf_const[4][0], max = vdf_const[4][1])
            p_vdf.add('u_par_b_par', r_vdf.params['u_par_b_par'], vary = vary_list[5], min = vdf_const[5][0], max = vdf_const[5][1])
            p_vdf.add('v_th_par_b_par', r_vdf.params['v_th_par_b_par'], vary = vary_list[6], min = vdf_const[6][0], max = vdf_const[6][1])
            p_vdf.add('v_th_perp_b_par', r_vdf.params['v_th_perp_b_par'], vary = vary_list[7], min = vdf_const[7][0], max = vdf_const[7][1])
            p_vdf.add('kappa_b_par', r_vdf.params['kappa_b_par'], vary = vary_list[8], min = vdf_const[8][0], max = vdf_const[8][1])
        else :
            p_vdf.add('n_b_par', 0.0, vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
            p_vdf.add('u_par_b_par', 0.0, vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
            p_vdf.add('v_th_par_b_par', 1.0, vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
            p_vdf.add('v_th_perp_b_par', 1.0, vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
            p_vdf.add('kappa_b_par', 10, vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
        if r_beam_anti_par != None :
            p_vdf.add('n_b_anti_par', r_vdf.params['n_b_anti_par'], vary = vary_list[9], min = vdf_const[9][0], max = vdf_const[9][1])
            p_vdf.add('u_par_b_anti_par', r_vdf.params['u_par_b_anti_par'], vary = vary_list[10], min = vdf_const[10][0], max = vdf_const[10][1])
            p_vdf.add('v_th_par_b_anti_par', r_vdf.params['v_th_par_b_anti_par'], vary = vary_list[11], min = vdf_const[11][0], max = vdf_const[11][1])
            p_vdf.add('v_th_perp_b_anti_par', r_vdf.params['v_th_perp_b_anti_par'], vary = vary_list[12], min = vdf_const[12][0], max = vdf_const[12][1])
            p_vdf.add('kappa_b_anti_par', r_vdf.params['kappa_b_anti_par'], vary = vary_list[13], min = vdf_const[13][0], max = vdf_const[13][1])
        else :
            p_vdf.add('n_b_anti_par', 0.0, vary = False, min = vdf_const[9][0], max = vdf_const[9][1])
            p_vdf.add('u_par_b_anti_par', 0.0, vary = False, min = vdf_const[10][0], max = vdf_const[10][1])
            p_vdf.add('v_th_par_b_anti_par', 1.0, vary = False, min = vdf_const[11][0], max = vdf_const[11][1])
            p_vdf.add('v_th_perp_b_anti_par', 1.0, vary = False, min = vdf_const[12][0], max = vdf_const[12][1])
            p_vdf.add('kappa_b_anti_par', 10, vary = False, min = vdf_const[13][0], max = vdf_const[13][1])
        p_vdf.add('n_h', r_vdf.params['n_h'], vary = vary_list[14], min = vdf_const[14][0], max = vdf_const[14][1])
        p_vdf.add('u_par_h', r_vdf.params['u_par_h'], vary = vary_list[15], min = vdf_const[15][0], max = vdf_const[15][1], expr = '(u_par_c*n_c + u_par_b_par*n_b_par + u_par_b_anti_par*n_b_anti_par)/n_h')
        p_vdf.add('v_th_par_h', r_vdf.params['v_th_par_h'], vary = vary_list[16], min = vdf_const[16][0], max = vdf_const[16][1])
        p_vdf.add('v_th_perp_h', r_vdf.params['v_th_perp_h'], vary = vary_list[17], min = vdf_const[17][0], max = vdf_const[17][1])
        p_vdf.add('kappa', r_vdf.params['kappa'], vary = vary_list[18], min = vdf_const[18][0], max = vdf_const[18][1])

        try :
            r_vdf = vdf_2beam.fit(psd_data, p_vdf, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
        except ValueError :
            # print('2 vdf_1beam dint fit')
            r_vdf = None

        return r_vdf

    vary_list = [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]]

    for ii in range(len(vary_list)) :
        if r_vdf != None :
            r_vdf = fit_update(p_vdf, r_vdf, vary_list[ii], psd_data, v_par, v_perp, vdf_const)
        else :
            print('final fit is not working')

    if r_vdf != None :
        fit_vdf = vdf_2beam.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    else:
        fit_vdf = vdf_2beam.eval(p_vdf, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_vdf, fit_vdf

def vdf_fit_function_non_beam(psd_data, v_par, v_perp, n, r_core, r_beam_params, r_halo) :

    vdf_const = []

    ''' use standard errors as bounds for these varibales '''
    vdf_const.append([-1*r_core.params['n'].stderr + r_core.params['n'].value, r_core.params['n'].stderr + r_core.params['n'].value])
    vdf_const.append([-1*r_core.params['u_par'].stderr + r_core.params['u_par'].value, r_core.params['u_par'].stderr + r_core.params['u_par'].value])
    vdf_const.append([-1*r_core.params['v_th_par'].stderr + r_core.params['v_th_par'].value, r_core.params['v_th_par'].stderr + r_core.params['v_th_par'].value])
    vdf_const.append([-1*r_core.params['v_th_perp'].stderr + r_core.params['v_th_perp'].value, r_core.params['v_th_perp'].stderr + r_core.params['v_th_perp'].value])
    vdf_const.append([0.0, 1.0])
    vdf_const.append([0.0, 1.0])
    vdf_const.append([0.0, 1.0])
    vdf_const.append([0.0, 1.0])
    vdf_const.append([0.0, 1.0])
    vdf_const.append([-1*r_halo.params['n'].stderr + r_halo.params['n'].value, r_halo.params['n'].stderr + r_halo.params['n'].value])
    vdf_const.append([-1*r_halo.params['u_par'].stderr + r_halo.params['u_par'].value, r_halo.params['u_par'].stderr + r_halo.params['u_par'].value])
    vdf_const.append([-1*r_halo.params['v_th_par'].stderr + r_halo.params['v_th_par'].value, r_halo.params['v_th_par'].stderr + r_halo.params['v_th_par'].value])
    vdf_const.append([-1*r_halo.params['v_th_perp'].stderr + r_halo.params['v_th_perp'].value, r_halo.params['v_th_perp'].stderr + r_halo.params['v_th_perp'].value])
    vdf_const.append([-1*r_halo.params['kappa'].stderr + r_halo.params['kappa'].value, r_halo.params['kappa'].stderr + r_halo.params['kappa'].value])

    for j in range(2) : vdf_const.append([None, None])

    p_vdf = lmfit.Parameters()
    p_vdf.add('n_c', r_core.params['n'], vary = False, min = vdf_const[0][0], max = vdf_const[0][1])
    p_vdf.add('u_par_c', r_core.params['u_par'], vary = False, min = vdf_const[1][0], max = vdf_const[1][1])
    p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary = False, min = vdf_const[2][0], max = vdf_const[2][1])
    p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary = False, min = vdf_const[3][0], max = vdf_const[3][1])
    p_vdf.add('n_b', 0.0, vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
    p_vdf.add('u_par_b', 0.0, vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
    p_vdf.add('v_th_par_b', 1.0, vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
    p_vdf.add('v_th_perp_b', 1.0, vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
    p_vdf.add('kappa_b', 1.0, vary = False, min = vdf_const[8][0], max = vdf_const[8][1])
    p_vdf.add('n_h', r_halo.params['n'], vary = False, min = vdf_const[9][0], max = vdf_const[9][1])
    p_vdf.add('u_par_h', r_halo.params['u_par'], vary = False, min = vdf_const[10][0], max = vdf_const[10][1], expr = '(u_par_c*n_c)/n_h')
    p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary = False, min = vdf_const[11][0], max = vdf_const[11][1])
    p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary = False, min = vdf_const[12][0], max = vdf_const[12][1])
    p_vdf.add('kappa', r_halo.params['kappa'], vary = False, min = vdf_const[13][0], max = vdf_const[13][1])
    p_vdf.add('n_sc', n, vary = False, min = vdf_const[14][0], max = vdf_const[14][1])

    try :
        r_vdf = vdf_1beam.fit(psd_data, p_vdf, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
    except (ValueError, TypeError) :
        # print('1 vdf_1beam dint fit')
        r_vdf = None

    def fit_update(p_vdf, r_vdf, vary_list, psd_data, v_par, v_perp, vdf_const) :

        p_vdf.add('n_c', r_vdf.params['n_c'], vary = vary_list[0], min = vdf_const[0][0], max = vdf_const[0][1])
        p_vdf.add('u_par_c', r_vdf.params['u_par_c'], vary = vary_list[1], min = vdf_const[1][0], max = vdf_const[1][1])
        p_vdf.add('v_th_par_c', r_vdf.params['v_th_par_c'], vary = vary_list[2], min = vdf_const[2][0], max = vdf_const[2][1])
        p_vdf.add('v_th_perp_c', r_vdf.params['v_th_perp_c'], vary = vary_list[3], min = vdf_const[3][0], max = vdf_const[3][1])
        p_vdf.add('n_b', 0.0, vary = False, min = vdf_const[4][0], max = vdf_const[4][1])
        p_vdf.add('u_par_b', 0.0, vary = False, min = vdf_const[5][0], max = vdf_const[5][1])
        p_vdf.add('v_th_par_b', 1.0, vary = False, min = vdf_const[6][0], max = vdf_const[6][1])
        p_vdf.add('v_th_perp_b', 1.0, vary = False, min = vdf_const[7][0], max = vdf_const[7][1])
        p_vdf.add('n_h', r_vdf.params['n_h'], vary = vary_list[8], min = vdf_const[8][0], max = vdf_const[8][1])
        p_vdf.add('u_par_h', r_vdf.params['u_par_h'], vary = vary_list[9], min = vdf_const[9][0], max = vdf_const[9][1], expr = '(u_par_c*n_c)/n_h')
        p_vdf.add('v_th_par_h', r_vdf.params['v_th_par_h'], vary = vary_list[10], min = vdf_const[10][0], max = vdf_const[10][1])
        p_vdf.add('v_th_perp_h', r_vdf.params['v_th_perp_h'], vary = vary_list[11], min = vdf_const[11][0], max = vdf_const[11][1])
        p_vdf.add('kappa', r_vdf.params['kappa'], vary = vary_list[12], min = vdf_const[12][0], max = vdf_const[12][1])

        try :
            r_vdf = vdf_1beam.fit(psd_data, p_vdf, method = method, v_par = v_par, v_perp = v_perp, max_nfev=300)
        except ValueError :
            # print('2 vdf_1beam dint fit')
            r_vdf = None

        return r_vdf

    vary_list = [[True, True, True, True, False, False, False, False, False, True, True, True, True, True, False]]

    for ii in range(len(vary_list)) :
        if r_vdf != None :
            r_vdf = fit_update(p_vdf, r_vdf, vary_list[ii], psd_data, v_par, v_perp, vdf_const)
        else :
            print('final fit is not working')

    if r_vdf != None :
        # print(r_vdf.chisqr)
        fit_vdf = vdf_1beam.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    else:
        fit_vdf = vdf_1beam.eval(p_vdf, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_vdf, fit_vdf

def plot_vdf_pieces() :

    ''' plot results '''
    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    color_lines = color_lines(np.linspace(0, 1, 4 + 1))

    fig, ax = plt.subplots(2, 2, figsize = (8,8), constrained_layout=True)
    # fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

    ''' full data (not selected by halo, strahl, core energy ranges) '''
    j = 1
    temp = pad[:,-j,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.8 :
        j += 1
        temp = pad[:,-j-1,0]
    jj = 0
    temp = pad[:,jj,0]
    while len(temp[~np.isnan(temp)])/len(temp) < 0.8 :
        jj += 1
        temp = pad[:,jj,0]
    v_par_pa_idx, anti_v_par_pa_idx, v_perp_pa_idx = jj, -1*j, np.shape(pad_in)[2]//2
    ax[0,0].scatter(swa_energy, pad[:,0,0], s = s_set, color = '0.8')
    ax[0,0].scatter(-1*swa_energy, pad[:,-2,0], s = s_set, color = '0.8')
    ax[0,1].scatter(swa_energy, pad[:,-2,0], s = s_set, color = '0.8')

    ''' PARALLEL '''
    ''' SC electron '''
    ax[0,0].scatter(swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, v_par_pa_idx, 0], s = s_set, color = 'k', label = 'SC electrons')
    ax[0,0].scatter(-1*swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, anti_v_par_pa_idx,0], s = s_set, color = 'k')

    ''' core v_par > 0 '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_par_pa_idx,0], s = s_set, color = color_lines[0])

    ''' core v_par < 0 '''
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0], label = 'Core fit')
    ax[0,0].scatter(-1*swa_energy[high_c:low_c], pad[high_c:low_c,anti_v_par_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')

    if anti_par_strahl_cond == True :
        pa_upper_bound_idx = np.argmin(np.abs(pa_anti_par_bound - pitch_angles))
        ''' halo v_par > 0 '''
        ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
        ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx,0], s = s_set, color = color_lines[1])
        for j in range(1,10) :
            ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_par_pa_idx+j,0], s = s_set, color = color_lines[1])

        ''' halo v_par < 0 '''
        ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1], label = 'Halo fit')
        ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx,0], s = s_set, color = color_lines[1], label = 'Halo')
        for j in range(1,10) :
            ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,pa_upper_bound_idx-j,0], s = s_set, color = color_lines[1])

        ''' beam '''
        ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_beam[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Beam fit')
        ax[0,0].scatter(-1*swa_energy[high_b:low_b], pad[high_b:low_b,anti_v_par_pa_idx,0], s = s_set, color = color_lines[2], label = 'Beam')
        for j in range(1,3) :
            ax[0,0].scatter(-1*swa_energy[high_b:low_b], pad[high_b:low_b,anti_v_par_pa_idx-j,0], s = s_set, color = color_lines[2], label = 'Beam')

    ''' full model vdf_1beam '''
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3], label = 'Model')
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)

    ax[0,0].set_xlabel('Energy [eV]')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0,0].set_ylim(np.max(pad[:,2,0])/(10**8), 2*np.max(pad[:,2,0]))

    ax[0,0].legend(loc = 8)

    ticks = ax[0,0].get_xticks()
    ticks = np.delete(ticks, (3,5))
    ax[0,0].set_xticks(ticks)


    ''' PERP '''
    ''' SC electron '''
    ax[0,1].scatter(swa_energy[high_sc:low_sc], pad[high_sc:low_sc,v_perp_pa_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    ''' core v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), fit_core[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[0], label = 'Core fit')
    ax[0,1].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_perp_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')

    ''' halo v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), fit_halo[:,np.shape(fit_core)[1]//2], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2,0], s = s_set, color = color_lines[1], label = 'Halo')
    for j in range(1,5) :
        ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx-2+j,0], s = s_set, color = color_lines[1])

    ''' full vdf_1beam '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:,np.shape(v_perp_mesh)[1]//2]), fit_vdf[:,np.shape(fit_core)[1]//2], lw = lw_set*1.5, ls = ':', color = color_lines[3], label = 'Model')

    ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')

    ax[0,1].set_xlabel('Energy [eV]')
    ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')

    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(10**8), 2*np.max(pad[:,2,0]))

    ''' Pitch angle fit '''
    ax[1,0].scatter(pitch_angle_centers, mean_energy_pad, s = s_set, color = color_lines[0], label = 'VDF averaged over energy')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[1,0].set_xlabel('Pitch angle [deg.]')

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])

    ax[1,0].legend()

    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0]
    paw_0_mask = (pad_params_energy[:,2] > 0)
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)
    # print(swa_energy[paw_180 < 0])
    # print('1', np.argmin(np.abs(swa_energy[paw_180 < 0][0] - swa_energy)))
    # ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    # ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    if r_beam_params != None : ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')

    ax[1,1].set_ylabel('Pitch angle width [deg.]')
    ax[1,1].set_xlabel('Energy [eV]')

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()


    plt.savefig(plot_path+"fit_demo_pieces_"+str(i)+"_.pdf", format='pdf', bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

''' End of functions '''

''' initial constraints '''
method = 'leastsq'

''' set spacecraft electron energy limits '''
low_sc_val, high_sc_val = 0, 12
low_sc, high_sc = np.argmin(np.abs(low_sc_val - swa_energy)), np.argmin(np.abs(high_sc_val - swa_energy))

''' remaining limits are set time locally '''

''' set halo energy limits '''
low_h_val, high_h_val = 100, 2000
low_h, high_h = np.argmin(np.abs(low_h_val - swa_energy)), np.argmin(np.abs(high_h_val - swa_energy))
n_halo_fact = 10 # reduce SC density for initial guess
n_beam_fact = 50 # reduce SC density for initial guess

''' set PAD energy limits for the energy average over PSD before fit '''
pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
low_pad_val, high_pad_val = 61, 1000
low_pad, high_pad = np.argmin(np.abs(low_pad_val - swa_energy)), np.argmin(np.abs(high_pad_val - swa_energy))
PAW_coeff = 2*np.sqrt(2*np.log(2))
''' velocity spans for the vdf_1beams '''
v_par_arr, v_perp_arr = np.linspace(-1*np.max(eV_to_vel(swa_energy)), np.max(eV_to_vel(swa_energy)), 1000), np.linspace(eV_to_vel(swa_energy)[0], 0.0, 1000)
v_par_mesh, v_perp_mesh = np.meshgrid(v_par_arr, v_perp_arr)

''' Kappa density factor, this uses D from the fit functions which is 10  '''
n_kappa_factor = (2*np.sqrt(10))/np.sqrt(10+1)

''' Important parameters '''
beam_ratio_cond = 1.2
deficit_ratio_cond = 0.5
kappa_b_init = 10

beg, end = 0, len(epoch_EAS)
# beg, end = 0, 1

''' some additional things to set '''
stderr_fact = 0.05 # what percent of the value to set the standard error to if the error analysis fails
n_const_range = [0.8, 1.2]
u_const_range = [0.8, 1.2]
core_init = [n_sc[beg], 0, 10**8, 10**8]

''' Make plots? '''
cond_final_plot, cond_pad_plot = False, False

vdf_save, core_save, halo_save, beam_par_save, beam_anti_par_save, pad_save = [], [], [], [], [], []

for i in tqdm(range(beg, end)) :
# for i in tqdm(range(1)) :

    ''' set how often to output plots '''
    if i % 2 == 0 :
        cond_final_plot, cond_pad_plot = True, False
    else :
        cond_final_plot, cond_pad_plot = False, False

    ''' set data for this loop '''
    n_in = n_sc[i]
    pad = pad_in[i,:,:,:]
    pa_vec = pa_vec_in[i,:,:,:]

    ''' take average then fit '''
    # initial, final = 7, 10
    # n_in = np.mean(n_sc[initial:final])
    # pad = np.mean(pad_in[initial:final,:,:,:], axis = 0)
    # pa_vec = np.mean(pa_vec_in[initial:final,:,:,:], axis = 0)

    ''' first two are energy ranges second 3 are the result object '''
    low_c, high_c = None, None
    low_b, high_b = None, None
    r_beam_params = None
    r_halo = None
    r_core = None

    ''' define all the constraints'''
    ''' core '''
    '''param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp']'''
    core_const = [[None, None], [None, None], [None, None], [None, None]]

    ''' beam '''
    '''param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp']'''
    beam_const = [[0, 0.9*n_in], [None, None], [None, None], [None, None], [None, None]]
    ''' beam conditions have to be set after beam direciton is determined '''

    ''' Halo fit '''
    '''param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa']'''
    halo_const = [[0, n_in/10], [None, None], [None, None], [None, None], [None, None]]
    halo_init = [n_in/n_halo_fact, 0, (10**9)/2, (10**9)/2, 5]

    if np.isnan(pa_vec[:,:,0]).all() == False :

        '''  PAD fit '''
        mean_energy_pad = np.nanmean(pad[high_pad:low_pad,:,0], axis = 0) # mean over energy range
        std_energy_pad = np.nanstd(pad[high_pad:low_pad,:,0], axis = 0)/np.sqrt(np.shape(pad[high_pad:low_pad,:,0])[0])
        mean_pad_params, mean_fit_pad = pad_fit_function(mean_energy_pad, pitch_angle_centers, pitch_angles)

        ''' compute strahl and deficit conditions '''
        par_strahl_cond, anti_par_strahl_cond, par_def_cond, anti_par_def_cond = False, False, False, False
        if (mean_pad_params['P_0']/mean_pad_params['P_B']) + 1 > beam_ratio_cond :
            par_strahl_cond = True
        if mean_pad_params['P_0']/mean_pad_params['P_B'] < deficit_ratio_cond :
            par_def_cond = True
        if (mean_pad_params['P_180']/mean_pad_params['P_B']) + 1 > beam_ratio_cond :
            anti_par_strahl_cond = True
        if mean_pad_params['P_180']/mean_pad_params['P_B'] < deficit_ratio_cond :
            anti_par_def_cond = True

        ''' fit PAD at each energy '''
        pad_params_energy, pad_fit_energy = pad_fit_operation()

        ''' Set energy ranges strahl and halo '''

        if anti_par_strahl_cond == True and par_strahl_cond == False :
            ''' this is an attempt at setting beam energy limits for fits '''
            # max_thresh = ( PAW_coeff*np.nanmax(pad_params_energy[:,1]) )/2 # half the largest PAW
            # mask = (pad_params_energy[:,1] < max_thresh) & (swa_energy > 20) & (swa_energy < 1000) # mask based on half largest and appropriate energies
            # anti_par_beam_paw_thresh = PAW_coeff*np.nanmean(pad_params_energy[mask,1]) # average of non-masked
            # mask = np.full(len(pad_params_energy[:,1]), True) # all true
            # mask[np.nanargmin(np.abs(2*max_thresh - PAW_coeff*pad_params_energy[:,1]))+1:] = False # Change PAWs before largest to false
            # temp_arg = np.sum(np.argwhere(PAW_coeff*pad_params_energy[mask,1] < anti_par_beam_paw_thresh), axis = 1)[-1] # the last arugment less than the beam threshold are, before the peak
            # anti_par_beam_energy_thresh = swa_energy[np.nanargmin(np.abs(pad_params_energy[mask,1][temp_arg] - pad_params_energy[:,1]))] # the energy at which the first PAW greater than the peak is less than the threshld
            ''' set energy limits '''
            anti_par_beam_energy_thresh = 70

            ''' set beam energy limits '''
            low_b_val, high_b_val = anti_par_beam_energy_thresh, 500
            low_b, high_b = np.argmin(np.abs(low_b_val - swa_energy)), np.argmin(np.abs(high_b_val - swa_energy))

            # print(anti_par_beam_energy_thresh)

            ''' set core energy limits '''
            low_c_val, high_c_val = 14, anti_par_beam_energy_thresh
            low_c, high_c = np.argmin(np.abs(low_c_val - swa_energy)), np.argmin(np.abs(high_c_val - swa_energy))

        if anti_par_strahl_cond == False and par_strahl_cond == True :
            ''' this is an attempt at setting beam energy limits for fits '''
            # max_thresh = ( PAW_coeff*np.nanmax(pad_params_energy[:,1]) )/2 # half the largest PAW
            # mask = (pad_params_energy[:,1] < max_thresh) & (swa_energy > 20) & (swa_energy < 1000) # mask based on half largest and appropriate energies
            # anti_par_beam_paw_thresh = PAW_coeff*np.nanmean(pad_params_energy[mask,1]) # average of non-masked
            # mask = np.full(len(pad_params_energy[:,1]), True) # all true
            # mask[np.nanargmin(np.abs(2*max_thresh - PAW_coeff*pad_params_energy[:,1]))+1:] = False # Change PAWs before largest to false
            # temp_arg = np.sum(np.argwhere(PAW_coeff*pad_params_energy[mask,1] < anti_par_beam_paw_thresh), axis = 1)[-1] # the last arugment less than the beam threshold are, before the peak
            # anti_par_beam_energy_thresh = swa_energy[np.nanargmin(np.abs(pad_params_energy[mask,1][temp_arg] - pad_params_energy[:,1]))] # the energy at which the first PAW greater than the peak is less than the threshld

            ''' set energy limits '''
            par_beam_energy_thresh = 70

            ''' set beam energy limits '''
            low_b_val, high_b_val = par_beam_energy_thresh, 500
            low_b, high_b = np.argmin(np.abs(low_b_val - swa_energy)), np.argmin(np.abs(high_b_val - swa_energy))

            ''' set core energy limits '''
            low_c_val, high_c_val = 14, par_beam_energy_thresh
            low_c, high_c = np.argmin(np.abs(low_c_val - swa_energy)), np.argmin(np.abs(high_c_val - swa_energy))

        if anti_par_strahl_cond == True and par_strahl_cond == True :
            ''' determine the beam energy threshold '''
            anti_par_beam_energy_thresh = 70

            ''' set core energy limits '''
            low_c_val, high_c_val = 14, anti_par_beam_energy_thresh
            low_c, high_c = np.argmin(np.abs(low_c_val - swa_energy)), np.argmin(np.abs(high_c_val - swa_energy))

            ''' set beam energy limits '''
            low_b_val, high_b_val = anti_par_beam_energy_thresh, 500
            low_b, high_b = np.argmin(np.abs(low_b_val - swa_energy)), np.argmin(np.abs(high_b_val - swa_energy))

        if anti_par_strahl_cond == False and par_strahl_cond == False :
            ''' set core energy limits '''
            anti_par_beam_energy_thresh = 70
            low_c_val, high_c_val = 14, 70
            low_c, high_c = np.argmin(np.abs(low_c_val - swa_energy)), np.argmin(np.abs(high_c_val - swa_energy))

        ''' plot the PAD fits in energy '''
        if cond_pad_plot == True : pad_fit_plot()

        ''' parama_name = [n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b, u_par_b, v_th_par_b, v_th_perp_b, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa)'''

        ''' Core fit'''
        n_core_ratio = 0.8
        core_init[0] = n_in*n_core_ratio
        v_th_core_init = ((core_init[0]/((np.pi**(3/2))*np.nanmax(pa_vec[high_c:low_c,:,1])))**(1/3))
        core_init[2], core_init[3] = v_th_core_init, v_th_core_init

        ''' Set core velocity based on beam location '''
        if anti_par_strahl_cond == False and par_strahl_cond == False :
            core_init[1] = 0.0
        if anti_par_strahl_cond == False and par_strahl_cond == True :
            core_init[1] = -1*eV_to_vel(2)
            core_const = [[core_const[0][0], core_const[0][1]], [-np.inf, 0], [None, None], [None, None]]
            halo_init[1] = -1*eV_to_vel(2)
            halo_const = [[halo_const[0][0], halo_const[0][1]], [-np.inf, 0], [halo_const[2][0], halo_const[2][1]], [halo_const[3][0], halo_const[3][1]], [halo_const[4][0], halo_const[4][1]]]
        if anti_par_strahl_cond == True and par_strahl_cond == False :
            core_init[1] = eV_to_vel(2)
            core_const = [[core_const[0][0], core_const[0][1]], [0, np.inf], [None, None], [None, None]]
            halo_init[1] = eV_to_vel(2)
            halo_const = [[halo_const[0][0], halo_const[0][1]], [0, np.inf], [halo_const[2][0], halo_const[2][1]], [halo_const[3][0], halo_const[3][1]], [halo_const[4][0], halo_const[4][1]]]
        if anti_par_strahl_cond == True and par_strahl_cond == True :
            core_init[1] = 0.0

        r_core, fit_core = core_fit_function(pa_vec[high_c:low_c,:,1], pa_vec[high_c:low_c,:,2], pa_vec[high_c:low_c,:,3], core_const, core_init)

        ''' Fit halo and beam (if there is a beam) '''
        ''' anti par beam '''
        if anti_par_strahl_cond == True and par_strahl_cond == False :

            r_beam_par = None

            ''' beam '''
            '''param_names = ['n', 'u_par', 'u_perp', 'v_th_par', 'v_th_perp']'''
            beam_const = [[beam_const[0][0], beam_const[0][1]], [None, 0], [None, None], [None, None], [None, None]]
            ''' beam conditions have to be set after beam direciton is determined '''
            ''' [pa, psd, v_par, v_perp] '''
            pa_anti_par_bound = 180 - mean_pad_params['W_180']*PAW_coeff # use mean energy PAW
            temp_b_pa, temp_b_psd, temp_b_v_par, temp_b_v_perp, temp_b_c = np.ravel(pa_vec[high_b:low_b,:,0]), np.ravel(pa_vec[high_b:low_b,:,1]), np.ravel(pa_vec[high_b:low_b,:,2]), np.ravel(pa_vec[high_b:low_b,:,3]), np.ravel(pa_vec[high_b:low_b,:,4])
            temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp = np.ravel(pa_vec[high_h:low_h,:,0]), np.ravel(pa_vec[high_h:low_h,:,1]), np.ravel(pa_vec[high_h:low_h,:,2]), np.ravel(pa_vec[high_h:low_h,:,3])

            if pa_anti_par_bound < 90 :
                r_halo, fit_halo = halo_fit_function(temp_h_psd[temp_h_pa < pa_anti_par_bound], temp_h_v_par[temp_h_pa < pa_anti_par_bound], temp_h_v_perp[temp_h_pa < pa_anti_par_bound], halo_const, halo_init)
            else :
                pa_anti_par_bound = 90
                r_halo, fit_halo = halo_fit_function(temp_h_psd[temp_h_pa < pa_anti_par_bound], temp_h_v_par[temp_h_pa < pa_anti_par_bound], temp_h_v_perp[temp_h_pa < pa_anti_par_bound], halo_const, halo_init)

            ''' this finds the actual superthermal density for claculation of the beam velocity '''
            if r_halo != None :
                fit_halo_hallow = halo_hallow.eval(r_halo.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
                halo_real_n = int_vdf(fit_halo_hallow, v_par_arr, v_perp_arr)/r_halo.params['n'].value
            else :
                halo_real_n = 0.15

            ''' guess based on constraints '''
            n_beam_init = n_in - r_core.params['n'].value - halo_real_n # this vlaue corrects for the halo over estiamte of the density
            u_beam_init = (r_core.params['n'].value*r_core.params['u_par'].value - halo_real_n*r_halo.params['u_par'].value)/n_beam_init
            u_beam_init = -1*eV_to_vel(70)

            if pa_anti_par_bound < 10 :
                pa_anti_par_bound = 170

            v_th_b_init = n_kappa_factor*(((n_beam_init)/((np.pi**(3/2))*np.max(temp_b_psd[temp_b_pa > pa_anti_par_bound])))**(1/3))

            # beam_init = [(n_beam_init)/n_kappa_factor, u_beam_init, v_th_b_init, v_th_b_init, kappa_b_init]

            print('beam initial conditions are static, turn off')
            beam_init = [(3.3*n_beam_init)/n_kappa_factor, 0.90*u_beam_init, 1.6*v_th_b_init, 0.8*v_th_b_init, 2*kappa_b_init]


            plot_cond_beam = False
            beam_const[0][0], beam_const[0][1] = beam_init[0]*n_const_range[0], beam_init[0]*n_const_range[1]
            beam_const[1][0], beam_const[1][1] = u_beam_init*u_const_range[0], u_beam_init*u_const_range[1]
            # beam_const[2][0], beam_const[2][1] = v_th_b_init*n_const_range[0], v_th_b_init*n_const_range[1]
            # beam_const[3][0], beam_const[3][1] = v_th_b_init*n_const_range[0], v_th_b_init*n_const_range[1]

            r_beam_anti_par, fit_beam_anti_par = beam_fit_function(temp_b_psd[temp_b_pa > pa_anti_par_bound], temp_b_v_par[temp_b_pa > pa_anti_par_bound], temp_b_v_perp[temp_b_pa > pa_anti_par_bound], temp_b_c[temp_b_pa > pa_anti_par_bound], beam_const, beam_init)

        ''' par beam '''
        if anti_par_strahl_cond == False and par_strahl_cond == True :

            r_beam_anti_par = None

            ''' beam '''
            '''param_names = ['n', 'u_par', 'u_perp', 'v_th_par', 'v_th_perp']'''
            beam_const = [[beam_const[0][0], beam_const[0][1]], [None, 0], [None, None], [None, None], [None, None]]
            ''' beam conditions have to be set after beam direciton is determined '''
            ''' [pa, psd, v_par, v_perp] '''
            pa_par_bound = mean_pad_params['W_0']*PAW_coeff # use mean energy PAW
            temp_b_pa, temp_b_psd, temp_b_v_par, temp_b_v_perp, temp_b_c = np.ravel(pa_vec[high_b:low_b,:,0]), np.ravel(pa_vec[high_b:low_b,:,1]), np.ravel(pa_vec[high_b:low_b,:,2]), np.ravel(pa_vec[high_b:low_b,:,3]), np.ravel(pa_vec[high_b:low_b,:,4])
            temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp = np.ravel(pa_vec[high_h:low_h,:,0]), np.ravel(pa_vec[high_h:low_h,:,1]), np.ravel(pa_vec[high_h:low_h,:,2]), np.ravel(pa_vec[high_h:low_h,:,3])

            r_halo, fit_halo = halo_fit_function(temp_h_psd[temp_h_pa > pa_par_bound], temp_h_v_par[temp_h_pa > pa_par_bound], temp_h_v_perp[temp_h_pa > pa_par_bound], halo_const, halo_init)
            ''' this finds the actual superthermal density for claculation of the beam velocity '''
            if r_halo != None :
                fit_halo_hallow = halo_hallow.eval(r_halo.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
                halo_real_n = int_vdf(fit_halo_hallow, v_par_arr, v_perp_arr)/r_halo.params['n'].value
            else :
                halo_real_n = 0.15

            ''' guess based on constraints '''
            n_beam_init = n_in - r_core.params['n'].value - halo_real_n # this vlaue corrects for the halo over estiamte of the density
            u_beam_init = (r_core.params['n'].value*r_core.params['u_par'].value - halo_real_n*r_halo.params['u_par'].value)/n_beam_init
            u_beam_init = eV_to_vel(70)

            if pa_par_bound < 10 :
                pa_par_bound = 10

            v_th_b_init = n_kappa_factor*((n_beam_init/((np.pi**(3/2))*np.max(temp_b_psd[temp_b_pa < pa_par_bound])))**(1/3))
            beam_init = [n_beam_init/n_kappa_factor, u_beam_init, v_th_b_init, v_th_b_init, kappa_b_init]

            plot_cond_beam = False
            beam_const[0][0], beam_const[0][1] = beam_init[0]*n_const_range[0], beam_init[0]*n_const_range[1]
            beam_const[1][0], beam_const[1][1] = u_beam_init*u_const_range[0], u_beam_init*u_const_range[1]
            beam_const[2][0], beam_const[2][1] = v_th_b_init*n_const_range[0], v_th_b_init*n_const_range[1]
            beam_const[3][0], beam_const[3][1] = v_th_b_init*n_const_range[0], v_th_b_init*n_const_range[1]

            r_beam_par, fit_beam_par = beam_fit_function(temp_b_psd[temp_b_pa < pa_par_bound], temp_b_v_par[temp_b_pa < pa_par_bound], temp_b_v_perp[temp_b_pa < pa_par_bound], temp_b_c[temp_b_pa < pa_par_bound], beam_const, beam_init)

        ''' No beams '''
        if anti_par_strahl_cond == False and par_strahl_cond == False :

            r_beam_anti_par, r_beam_par = None, None

            temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp = np.ravel(pa_vec[high_h:low_h,:,0]), np.ravel(pa_vec[high_h:low_h,:,1]), np.ravel(pa_vec[high_h:low_h,:,2]), np.ravel(pa_vec[high_h:low_h,:,3])

            r_halo, fit_halo = halo_fit_function(temp_h_psd, temp_h_v_par, temp_h_v_perp, halo_const, halo_init)
            ''' this finds the actual superthermal density for claculation of the beam velocity '''
            if r_halo != None :
                fit_halo_hallow = halo_hallow.eval(r_halo.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
                halo_real_n = int_vdf(fit_halo_hallow, v_par_arr, v_perp_arr)/r_halo.params['n'].value
            else :
                halo_real_n = 0.15

        ''' Double beam routine '''
        if anti_par_strahl_cond == True and par_strahl_cond == True :

            ''' beam '''
            '''param_names = ['n', 'u_par', 'u_perp', 'v_th_par', 'v_th_perp']'''
            beam_const_par = [[beam_const[0][0], beam_const[0][1]], [None, 0], [None, None], [None, None], [None, None]]
            beam_const_anti_par = [[beam_const[0][0], beam_const[0][1]], [None, 0], [None, None], [None, None], [None, None]]

            ''' beam conditions have to be set after beam direciton is determined '''
            ''' [pa, psd, v_par, v_perp] '''
            pa_anti_par_bound = 180 - mean_pad_params['W_180']*PAW_coeff # use mean energy PAW
            pa_par_bound = mean_pad_params['W_0']*PAW_coeff # use mean energy PAW
            temp_b_pa, temp_b_psd, temp_b_v_par, temp_b_v_perp, temp_b_c = np.ravel(pa_vec[high_b:low_b,:,0]), np.ravel(pa_vec[high_b:low_b,:,1]), np.ravel(pa_vec[high_b:low_b,:,2]), np.ravel(pa_vec[high_b:low_b,:,3]), np.ravel(pa_vec[high_b:low_b,:,4])
            temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp = np.ravel(pa_vec[high_h:low_h,:,0]), np.ravel(pa_vec[high_h:low_h,:,1]), np.ravel(pa_vec[high_h:low_h,:,2]), np.ravel(pa_vec[high_h:low_h,:,3])

            # r_halo, fit_halo = halo_log_fit_function(temp_h_psd[temp_h_pa < pa_upper_bound], temp_h_v_par[temp_h_pa < pa_upper_bound], temp_h_v_perp[temp_h_pa < pa_upper_bound], n_in/n_halo_fact, halo_const)
            r_halo, fit_halo = halo_fit_function(temp_h_psd[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)], temp_h_v_par[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)], temp_h_v_perp[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)], halo_const, halo_init)

            ''' this finds the actual superthermal density for claculation of the beam velocity '''
            if r_halo != None :
                fit_halo_hallow = halo_hallow.eval(r_halo.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
                halo_real_n = int_vdf(fit_halo_hallow, v_par_arr, v_perp_arr)/r_halo.params['n'].value
            else :
                halo_real_n = 0.15


            # pt(halo_real_n)
            ''' guess based on constraints '''
            n_beam_total = n_in - r_core.params['n'].value - halo_real_n
            n_beam_anti_par_init = n_beam_total*(mean_pad_params['P_180']/(mean_pad_params['P_0'] + mean_pad_params['P_180']))
            n_beam_par_init = n_beam_total - n_beam_anti_par_init
            n_beam_anti_par_init = n_beam_total/2
            n_beam_par_init = n_beam_total/2
            ''' uses anti_Par_beam_energy_threshold to determine velocity ignores current condition '''
            u_beam_anti_par_init = -1*eV_to_vel(70)
            u_beam_par_init = eV_to_vel(70)

            if pa_par_bound < 10 :
                pa_par_bound = 10
            if pa_anti_par_bound > 170 :
                pa_anti_par_bound = 170

            v_th_b_par_init = n_kappa_factor*(((n_beam_par_init)/((np.pi**(3/2))*np.max(temp_b_psd[temp_b_pa < pa_par_bound])))**(1/3))
            v_th_b_anti_par_init = n_kappa_factor*(((n_beam_anti_par_init)/((np.pi**(3/2))*np.max(temp_b_psd[temp_b_pa > pa_anti_par_bound])))**(1/3))

            beam_par_init = [n_beam_par_init/n_kappa_factor, u_beam_par_init, v_th_b_par_init, v_th_b_par_init, kappa_b_init]
            beam_anti_par_init = [n_beam_anti_par_init/n_kappa_factor, u_beam_anti_par_init, v_th_b_anti_par_init, v_th_b_anti_par_init, kappa_b_init]

            plot_cond_beam = False
            beam_const_par[0][0], beam_const_par[0][1] = beam_par_init[0]*n_const_range[0], beam_par_init[0]*n_const_range[1]
            beam_const_par[1][0], beam_const_par[1][1] = u_beam_par_init*u_const_range[0], u_beam_par_init*u_const_range[1]
            beam_const_anti_par[0][0], beam_const_anti_par[0][1] = beam_anti_par_init[0]*n_const_range[0], beam_anti_par_init[0]*n_const_range[1]
            beam_const_anti_par[1][0], beam_const_anti_par[1][1] = u_beam_anti_par_init*u_const_range[0], u_beam_anti_par_init*u_const_range[1]

            r_beam_par, fit_beam_par = beam_fit_function(temp_b_psd[temp_b_pa < pa_par_bound], temp_b_v_par[temp_b_pa < pa_par_bound], temp_b_v_perp[temp_b_pa < pa_par_bound], temp_b_c[temp_b_pa < pa_par_bound], beam_const_par, beam_par_init)
            r_beam_anti_par, fit_beam_anti_par = beam_fit_function(temp_b_psd[temp_b_pa > pa_anti_par_bound], temp_b_v_par[temp_b_pa > pa_anti_par_bound], temp_b_v_perp[temp_b_pa > pa_anti_par_bound], temp_b_c[temp_b_pa > pa_anti_par_bound], beam_const_anti_par, beam_anti_par_init)


        ''' Final model fits with all the initial conditions from the individual fits '''
        if anti_par_strahl_cond == False and par_strahl_cond == False :
            r_vdf, fit_vdf = vdf_fit_function_non_beam(pa_vec[:low_c,:,1], pa_vec[:low_c,:,2], pa_vec[:low_c,:,3], n_in, r_core, r_beam_params, r_halo)
        if anti_par_strahl_cond == False and par_strahl_cond == True :
            r_vdf, fit_vdf = vdf_fit_function_one_beam(pa_vec[:low_c,:,1], pa_vec[:low_c,:,2], pa_vec[:low_c,:,3], n_in, r_core, r_beam_params, r_halo)
        if anti_par_strahl_cond == True and par_strahl_cond == False :
            r_vdf, fit_vdf = vdf_fit_function_one_beam(pa_vec[:low_c,:,1], pa_vec[:low_c,:,2], pa_vec[:low_c,:,3], n_in, r_core, r_beam_params, r_halo)
        if anti_par_strahl_cond == True and par_strahl_cond == True :
            r_vdf, fit_vdf = vdf_fit_function_two_beam(pa_vec[:low_c,:,1], pa_vec[:low_c,:,2], pa_vec[:low_c,:,3], n_in, r_core, r_beam_params, r_halo)

        ''' compute err prop chi-squared '''
        if r_vdf != None :
            red_chi = 0.0
            counter = 0
            if anti_par_strahl_cond + par_strahl_cond == 0 :
                for j in range(high_h, high_sc+1) :
                    for k in range(np.shape(pa_vec)[2]) :
                        if np.isnan(pa_vec[j,k,1]) == False :
                            if pa_vec[j,k,1] != 0.0 :
                                num = (pa_vec[j,k,1] - vdf_1beam.eval(r_vdf.params, v_par = pa_vec[j,k,2], v_perp = pa_vec[j,k,3]))**2
                                den = (pa_vec[j,k,1]/np.sqrt(pa_vec[j,k,4]))**2
                                red_chi += num/den
                                counter += 1
                            else :
                                red_chi += 0.0
                                counter += 1
                red_chi = red_chi/(counter - 9) # 9 is parameter number of fit.
            if anti_par_strahl_cond + par_strahl_cond == 1 :
                for j in range(high_h, high_sc+1) :
                    for k in range(np.shape(pa_vec)[2]) :
                        if np.isnan(pa_vec[j,k,1]) == False :
                            if pa_vec[j,k,1] != 0.0 :
                                num = (pa_vec[j,k,1] - vdf_1beam.eval(r_vdf.params, v_par = pa_vec[j,k,2], v_perp = pa_vec[j,k,3]))**2
                                den = (pa_vec[j,k,1]/np.sqrt(pa_vec[j,k,4]))**2
                                red_chi += num/den
                                counter += 1
                            else :
                                red_chi += 0.0
                                counter += 1
                red_chi = red_chi/(counter - 14) # 13 is parameter number of fit.
            if anti_par_strahl_cond + par_strahl_cond == 2 :
                for j in range(high_h, high_sc+1) :
                    for k in range(np.shape(pa_vec)[2]) :
                        if np.isnan(pa_vec[j,k,1]) == False :
                            if pa_vec[j,k,1] != 0.0 :
                                num = (pa_vec[j,k,1] - vdf_2beam.eval(r_vdf.params, v_par = pa_vec[j,k,2], v_perp = pa_vec[j,k,3]))**2
                                den = (pa_vec[j,k,1]/np.sqrt(pa_vec[j,k,4]))**2
                                red_chi += num/den
                                counter += 1
                            else :
                                red_chi += 0.0
                                counter += 1
                red_chi = red_chi/(counter - 19) # 17 is parameter number of fit.

        ''' Saving the full model parameters '''
        if r_vdf != None :
            r_vdf_keys = list(r_vdf.params.valuesdict().keys())
            value_list, stderr_list = [], []
            for j in range(len(r_vdf_keys)) :
                value_list.append(r_vdf.params[r_vdf_keys[j]].value)
                stderr_list.append(r_vdf.params[r_vdf_keys[j]].stderr)

            ''' Saving the final fitted data '''
            ''' ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c', 'n_b',
                    'u_par_b', 'v_th_par_b', 'v_th_perp_b', 'kappa_b',
                    'n_h', 'u_par_h', 'v_th_par_h', 'v_th_perp_h',
                    'kappa', 'u_p', 'n_sc']'''

            if anti_par_strahl_cond == True and par_strahl_cond == False :
                value_list = value_list[:4]+[np.nan, np.nan, np.nan, np.nan, np.nan]+value_list[4:]
                stderr_list = stderr_list[:4]+[np.nan, np.nan, np.nan, np.nan, np.nan]+stderr_list[4:]
            if anti_par_strahl_cond == False and par_strahl_cond == True :
                value_list = value_list[:9]+[np.nan, np.nan, np.nan, np.nan, np.nan]+value_list[9:]
                stderr_list = stderr_list[:9]+[np.nan, np.nan, np.nan, np.nan, np.nan]+stderr_list[9:]
            if anti_par_strahl_cond == False and par_strahl_cond == False :
                value_list = value_list[:4]+[np.nan, np.nan, np.nan, np.nan, np.nan]+value_list[4:]
                stderr_list = stderr_list[:4]+[np.nan, np.nan, np.nan, np.nan, np.nan]+stderr_list[4:]

            save_list = value_list+stderr_list+[r_vdf.aic, r_vdf.bic, r_vdf.chisqr, r_vdf.redchi, red_chi]
            print(len(save_list))
            vdf_save.append(save_list)
        else :
            vdf_save.append(45*[np.nan])

        ''' Saving the individual population fit data '''
        if r_core != None :
            r_core_keys = list(r_core.params.valuesdict().keys())
            value_list_core, stderr_list_core = [], []
            for j in range(len(r_core_keys)) :
                value_list_core.append(r_core.params[r_core_keys[j]].value)
                stderr_list_core.append(r_core.params[r_core_keys[j]].stderr)
            save_list = value_list_core+stderr_list_core+[r_core.aic, r_core.bic, r_core.chisqr, r_core.redchi]
            core_save.append(save_list)
        else :
            core_save.append(12*[np.nan])

        if r_beam_par != None :
            r_beam_par_keys = list(r_beam_par.params.valuesdict().keys())
            value_list_beam_par, stderr_list_beam_par = [], []
            for j in range(len(r_core_keys)) :
                value_list_beam_par.append(r_beam_par.params[r_beam_par_keys[j]].value)
                stderr_list_beam_par.append(r_core.params[r_beam_par_keys[j]].stderr)
            save_list = value_list_beam_par+stderr_list_beam_par+[r_beam_par.aic, r_beam_par.bic, r_beam_par.chisqr, r_beam_par.redchi]
            beam_par_save.append(save_list)
        else :
            beam_par_save.append(12*[np.nan])

        if r_beam_anti_par != None :
            r_beam_anti_par_keys = list(r_beam_anti_par.params.valuesdict().keys())
            value_list_beam_anti_par, stderr_list_beam_anti_par = [], []
            for j in range(len(r_core_keys)) :
                value_list_beam_anti_par.append(r_beam_anti_par.params[r_beam_anti_par_keys[j]].value)
                stderr_list_beam_anti_par.append(r_core.params[r_beam_anti_par_keys[j]].stderr)
            save_list = value_list_beam_anti_par+stderr_list_beam_anti_par+[r_beam_anti_par.aic, r_beam_anti_par.bic, r_beam_anti_par.chisqr, r_beam_anti_par.redchi]
            beam_anti_par_save.append(save_list)
        else :
            beam_anti_par_save.append(12*[np.nan])

        if r_halo != None :
            r_halo_keys = list(r_halo.params.valuesdict().keys())
            value_list_halo, stderr_list_halo = [], []
            for j in range(len(r_halo_keys)) :
                value_list_halo.append(r_halo.params[r_halo_keys[j]].value)
                stderr_list_halo.append(r_halo.params[r_halo_keys[j]].stderr)
            save_list = value_list_halo+stderr_list_halo+[r_halo.aic, r_halo.bic, r_halo.chisqr, r_halo.redchi]
            halo_save.append(save_list)
        else :
            halo_save.append(14*[np.nan])

        pad_save.append(pad_params_energy)

        ''' Plotting '''
        if cond_final_plot == True :
            if anti_par_strahl_cond + par_strahl_cond == 1 :
                plot_vdf_1beam_final()
            if anti_par_strahl_cond + par_strahl_cond == 2 :
                plot_vdf_2beam_final()
            if anti_par_strahl_cond + par_strahl_cond == 0 :
                plot_vdf_nobeam_final()

    else :

        pad_temp = []
        for j in range(len(swa_energy)) : pad_temp.append(5*[np.nan])
        pad_save.append(pad_temp)
        vdf_save.append(45*[np.nan])
        core_save.append(12*[np.nan])
        beam_par_save.append(12*[np.nan])
        beam_anti_par_save.append(12*[np.nan])
        halo_save.append(14*[np.nan])

pad_names = ['W_0', 'W_180', 'P_0', 'P_180', 'P_B']
vdf_name = ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c', 'n_b_par', 'u_par_b_par', 'v_th_par_b_par', 'v_th_perp_b_par', 'kappa_b_par', 'n_b_anti_par', 'u_par_b_anti_par', 'v_th_par_b_anti_par', 'v_th_perp_b_anti_par', 'kappa_b_par', 'n_h', 'u_par_h', 'v_th_par_h', 'v_th_perp_h', 'kappa', 'n_sc', 'sig n_c', 'sig u_par_c', 'sig v_th_par_c', 'sig v_th_perp_c', 'sig n_b_par', 'sig u_par_b_par', 'sig v_th_par_b_par', 'sig v_th_perp_b_par', 'sig kappa_b_par', 'sig n_b_anti_par', 'sig u_par_b_anti_par', 'sig v_th_par_b_anti_par', 'sig v_th_perp_b_anti_par', 'sig kappa_b_anti_par', 'sig n_h', 'sig u_par_h', 'sig v_th_par_h', 'sig v_th_perp_h', 'sig kappa', 'sig n_sc', 'Akaike', 'Bayesian', 'chisqr', 'redchi', 'err redchi']
core_name = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'sig n', 'sig u_par', 'sig v_th_par', 'sig v_th_perp', 'Akaike', 'Bayesian', 'chisqr', 'redchi']
beam_par_name = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'sig n', 'sig u_par', 'sig v_th_par', 'sig v_th_perp', 'Akaike', 'Bayesian', 'chisqr', 'redchi']
beam_anti_par_name = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'sig n', 'sig u_par', 'sig v_th_par', 'sig v_th_perp', 'Akaike', 'Bayesian', 'chisqr', 'redchi']
halo_name = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa', 'sig n', 'sig u_par', 'sig v_th_par', 'sig v_th_perp', 'sig kappa', 'Akaike', 'Bayesian', 'chisqr', 'redchi']



Path(data_save_path+"/"+file_name).mkdir(parents=True, exist_ok=True)

for i in range(len(pad_names)) : DataFrame(np.array(pad_save)[:,:,i]).to_pickle(data_save_path+"/"+file_name+"/"+'pad_save_'+pad_names[i])
DataFrame(vdf_save, columns = vdf_name).to_pickle(data_save_path+"/"+file_name+"/"+'vdf_save')
DataFrame(core_save, columns = core_name).to_pickle(data_save_path+"/"+file_name+"/"+'core_save')
DataFrame(beam_par_save, columns = beam_par_name).to_pickle(data_save_path+"/"+file_name+"/"+'par_beam_save')
DataFrame(beam_anti_par_save, columns = beam_anti_par_name).to_pickle(data_save_path+"/"+file_name+"/"+'anti_par_beam_save')
DataFrame(halo_save, columns = halo_name).to_pickle(data_save_path+"/"+file_name+"/"+'halo_save')

















''' end '''
