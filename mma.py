# James Parkin. Murray biocircuits lab. 2020.
# Attempt to create a suite of python functions and classes to aid in the 
# visualization and analysis of microscope movies on the HPC.

# TODO
# Convert notebook code for file accounting into functions

# OVERVIEW
# Module variables
# Functions

# Classes
# Acquisition
#   class associated with a single multi-D acquisition run using micromanager
#   initiated from only a directory that contains the acquisition images, it 
#   creates files_df.csv, timevec.npy, to facilitate analysis

import os
import sys
import time
import datetime
import re
import multiprocessing
from multiprocessing import Pool
import ast

import numpy as np
import matplotlib.colors as mpl_colors
from  matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
rc={'lines.linewidth': 2, 'axes.labelsize': 14, 'axes.titlesize': 14}

sns.set(rc=rc)

from scipy import ndimage as ndi
from scipy import signal
from scipy import optimize as opt
import scipy as scp
import pandas as pd

import matplotlib.animation as anm
from IPython.display import HTML

from skimage.external.tifffile import TiffWriter

from PIL import Image, UnidentifiedImageError

import skimage.filters
import skimage.io
import skimage.morphology
import skimage.external.tifffile

# Channel information for IX81 inverted microscope
all_channel_inds = [str(xx) for xx in range(6)]
all_channel_names = ['Brightfield', 'GFP', 'YFP', 'mScarlet', 'unk', 'sfCFP']
mpl_named_colors = ['gray', 'g', 'y', 'r', 'k', 'c']
all_channel_dict = dict(zip(all_channel_inds, all_channel_names))
objective_index = pd.Index(np.array(['5','4','2','1','0']),name='objective')
rel_mag = 1/np.array([1,1,10,6,10])
n_objs = len(rel_mag)
rel_bin = np.array([1,2,4])/4
n_bins = len(rel_bin)
rel_len_arr = np.tile(rel_mag.reshape((n_objs,1)),(1,n_bins))*np.tile(rel_bin.reshape((1,n_bins)),(n_objs,1))
binning_index = pd.Index(np.array(['1x1','2x2','4x4']),name='binning')
pixel_size_table = pd.DataFrame(index=objective_index, columns=binning_index, data=rel_len_arr*2.585)

# manual metadata
pad_bg_pos_lists = {'200626_finalday/img_1':(25,)}

pad_metadata_dict = {'180312_full_circuit_w_senders/20180302_2':((0,1,2)),
'180313_full_circuit_w_senders/20180313_1':((0,3,6),(1,4,7),(2,5,8)),
'180314_full_circuit_w_senders/20180314_1':(np.arange(0,3),np.arange(3,6),np.arange(6,9)),
'180316/20180316_fullcircuit_longpads_1':((0,1,2),),
'180320/20180320_fullcircuit_longpads_1':(np.arange(0,3),np.arange(3,6),np.arange(6,9)),
'180601/180601_1':((0,9,20,28),(29,30,31,1,2,3,4,5,6),(7,8,10,11,12,13,14,15,16,17,18,19,21,22,23,24),(25,26,27)),
'180711/20180711_1':((0,1),(8,7,6,5,4,3,2,32,31,30,29,28,27,23,12),(26,25,24,22,21,20,19,18,17,16,15,14,13,11,10,9)),
'181222_movie/20181222_1':((34,33,32,31,0,22,11),(35,36,1,2,3,4,5),(10,9,8,7,6),(12,13,14,15,16),(23,21,20,19,18,17),(24,25,26,27,28,29,30)),
'190126_rhl_strain_6/20190126_2':((0,1,7,8),(13,12,11,10,9),(14,2,3,4,5),(6,)),
'190126_rhl_strain_6/20190126_4':((1,12,23),(28,27,26,25,24),(6,7,8,9,10),(21,20,19,18,17,22),(11,13,14,15,16),(29,2,3,4,5)),
'190221_inducer_test/20190221_1':(np.arange(13),(13,31)),
'190222_pfb_test/20190222_1':(np.arange(57,88)),
'190725_osc2/data':(np.arange(0,8),np.arange(8,19),np.arange(19,32),np.arange(32,44),np.arange(44,56)),
'190726_i52_2/ladder_img_1':((0,1,2),(3,4,5),(6,7,8)),
'190727_pfbs/img_2':(np.arange(70,82),),
'180316/20180316_fullcircuit_longpads_1':((0,),(1,),(2,)),
'180316/20180316_fullcircuit_longpads_1am_1':((0,1),(2,3),(4,5)),
'180316/20180316_fullcircuit_longpads_2_3':((0,1,2),(3,4,5),(6,7,8)),
'20180319/20180319_fullcircuit_longpads_1':((0,1,2),(3,4,5),(6,7,8)),
#'movies/20180420_AGAIN_1':((0,),),
'20180509_three_fc_variants/20180509_1':((0,1,2),(3,4,5),(6,7,8)),
'20180517_i53_2_nosender/20180517_3':((0,1,2),(3,4,5,6,7,8),(9,10,11,12)),
'20181208_rhlpulsetry/20181208_1':((0,11,32,44,45,46,1,2,3,4,5,6,7,8,9,10),(12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30),(43,42,31,33,34,35,36,37,38,39,40,41)),
'20181208_rhlpulsetry/20181208_after_failure_1':((7,27,42,2,3,4,5,6),(33,32,31,30,29,25,26,28),(34,35,36,37,39,40,38),(43,44,41,0,1,47,46,45),(24,23,22,17,18,19,20,21),(8,9,10,11,12,13,14,15,16)),
'20181208_rhlpulsetry/20181208_fewwer_points_1':((2,3,4),(6,5),(7,8),(10,9,11,0,1)),
'movies/20181211_1':((0,),(1,6,7,8),(9,10,11,12,13),(2,3,4,5)),
'movies/20190108_1':((1,9),(15,16),(17,),(5,4,3,2,0),(14,13,12,11,10,9,8,6)),
'movies/20190108_again_1':((0,1,14),(21,22,23),(24,25,2,3,4),(5,6),(7,8,9,10,11),(20,19,18,17,16,15,13,12)),
'movies/20190110_both_pulsers_1':((0,10,40,19,43,42,41,34),(39,37,36,32,38,35,33,31),(30,27,24,29,28,26,25,23),(22,21,20,17,14,13,15,12,16,18),(11,6,8,5,4,7,9),(1,2,3,44)),
#'20190122_small_rhl_pads/20190122_1':((0,1,12,23),(47,2,3,4),(5,6,7,8),(10,11,9,14,13),(15,16,17,18),(22,21,20,19),(27,26,25,24),(28,29,30,31,32),(37,36,35,33),(38,39,40,41),(42,),(44,45,46,43,34)),
'20190122_small_rhl_pads/20190122_1':((0,14),(1,),(25,26,27,28),(2,3,4),(5,6),(7,8,9),(10,11,12,13),(15,16,17),(18,19),(20,21,22),(23,24,25),(26,27,28,29)),
'20190122_small_rhl_pads/20190122_b_1':((0,14),(1,),(25,26,27,28),(2,3,4),(5,6),(7,8,9),(10,11,12,13),(15,16,17),(18,19),(20,21,22),(23,24,25),(26,27,28,29))}

pad_mag_dict = {'180312_full_circuit_w_senders/20180302_2':True,
'180313_full_circuit_w_senders/20180313_1':True,
'180314_full_circuit_w_senders/20180314_1':True,
'180316/20180316_fullcircuit_longpads_1':True,
'180320/180320_fullcircuit_longpads_1':True,
'180601/180601_1':True,
'180711/20180711_1':True,
'180822_movie/20180822_1':True,
'181222_movie/20181222_1':True,
'190126_rhl_strain_6/20190126_2':True,
'190126_rhl_strain_6/20190126_4':True,
'190221_inducer_test/20190221_1':True,
'190222_pfb_test/20190222_1':True,
'190225_rhl_pulser/20190225_1':True,
'190226_rhl_pulsers_again/20190226_1':False,
'190226_rhl_pulsers_again/20190226_2_1':False,
'190303_cin_pulsers/20190303_3':True,
'190312_many_small_pads/20190312_1':True,
'190313_cin_again/20190313_3':False,
'190314_rhl_pulsers/20190314_1':False,
'190314_rhl_pulsers/20190314_2':False,
'190316/20190316_2':True,
'190411_constitutives_again/img_1':True,
'190413_constitutives_again/img_1':True,
'190418_receivers/img_2':False,
'190725_osc2/data':True,
'190726_i52_2/ladder_img_1':True,
'190727_pfbs/img_2':False,
'191018_rpa_senders_cin_pulse/img_1':False,
'191018_rpa_senders_cin_pulse/img_2':False,
'200613_unordered/img_3':True,
'200626_finalday/img_1':True,
'200627_secondfinalday/img_1':True,
'180316/20180316_fullcircuit_longpads_1':True,
'180316/20180316_fullcircuit_longpads_1am_1':True,
'180316/20180316_fullcircuit_longpads_2_3':True,
'20180319/20180319_fullcircuit_longpads_1':True,
'movies/20180420_AGAIN_1':True,
'20180509_three_fc_variants/20180509_1':True,
'20180517_i53_2_nosender/20180517_3':True,
'20181208_rhlpulsetry/20181208_1':True,
'20181208_rhlpulsetry/20181208_after_failure_1':True,
'20181208_rhlpulsetry/20181208_fewwer_points_1':True,
'movies/20181211_1':True,
'movies/20190108_1':True,
'movies/20190108_again_1':True,
'movies/20190110_both_pulsers_1':True,
'movies/20190224_1':True,
'20190122_small_rhl_pads/20190122_1':True,
'20190122_small_rhl_pads/20190122_b_1':True}

pad_obj_dict = {'180312_full_circuit_w_senders/20180302_2':'1',
'180313_full_circuit_w_senders/20180313_1':'1',
'180314_full_circuit_w_senders/20180314_1':'1',
'180316/20180316_fullcircuit_longpads_1':'1',
'180320/180320_fullcircuit_longpads_1':'1',
'180601/180601_1':'5',
'180711/20180711_1':'5',
'181222_movie/20181222_1':'5',
'190126_rhl_strain_6/20190126_2':'5',
'190126_rhl_strain_6/20190126_4':'5',
'190221_inducer_test/20190221_1':'5',
'190222_pfb_test/20190222_1':'5',
'190225_rhl_pulser/20190225_1':'5',
'190226_rhl_pulsers_again/20190226_1':'5',
'190226_rhl_pulsers_again/20190226_2_1':'5',
'190303_cin_pulsers/20190303_3':'5',
'190312_many_small_pads/20190312_1':'5',
'190313_cin_again/20190313_3':'5',
'190314_rhl_pulsers/20190314_1':'5',
'190314_rhl_pulsers/20190314_2':'5',
'190316/20190316_2':'5',
'190411_constitutives_again/img_1':'5',
'190413_constitutives_again/img_1':'5',
'190418_receivers/img_2':'5',
'190725_osc2/data':'5',
'190726_i52_2/ladder_img_1':'2',
'190727_pfbs/img_2':'5',
'191018_rpa_senders_cin_pulse/img_1':'5',
'191018_rpa_senders_cin_pulse/img_2':'5',
'191212_rpa_senders_again/bf_final_1':'5',
'191212_rpa_senders_again/img_1':'5',
'191217_rpa_thin_attmpt/img_1':'5',
'191219_thin_attempt_2/bf_initial':'5',
'191219_thin_attempt_2/img_3':'5',
'200130_cin_pulsers_and_reporters/img_1':'5',
'200131_cin_pulser_reporter/img_1':'5',
'200131_cin_pulser_reporter/img_2':'5',
'200213_cin_tests/bf_initial_1':'5',
'200213_cin_tests/img_1':'5',
'200307_prs/final_surplus_1':'5',
'200307_prs/img_1':'5',
'200307_prs/img_2':'5',
'200307_prs/img_3':'5',
'200307_prs/initial_1':'5',
'200307_prs/initial_2':'5',
'200309_prs/img_1':'5',
'200309_prs/img_2':'5',
'200309_prs/initial_bf_1':'5',
'200309_prs/sparse_grid_1':'5',
'200309_prs/sparse_grid_yfp_1':'5',
'200613_unordered/img_3':'5',
'200626_finalday/img_1':'5',
'200627_secondfinalday/img_1':'5',
'180316/20180316_fullcircuit_longpads_1':'5',
'180316/20180316_fullcircuit_longpads_1am_1':'5',
'180316/20180316_fullcircuit_longpads_2_3':'5',
'20180319/20180319_fullcircuit_longpads_1':'5',
'movies/20180420_AGAIN_1':'5',
'20180509_three_fc_variants/20180509_1':'5',
'20180517_i53_2_nosender/20180517_3':'5',
'20181208_rhlpulsetry/20181208_1':'5',
'20181208_rhlpulsetry/20181208_after_failure_1':'5',
'20181208_rhlpulsetry/20181208_fewwer_points_1':'5',
'movies/20181211_1':'5',
'movies/20190108_1':'5',
'movies/20190108_again_1':'5',
'movies/20190110_both_pulsers_1':'5',
'movies/20190224_1':'5',
'20190122_small_rhl_pads/20190122_1':'5',
'20190122_small_rhl_pads/20190122_b_1':'5'}

pad_rotation_dict = {'180312_full_circuit_w_senders/20180302_2':0,
'180313_full_circuit_w_senders/20180313_1':0,
'180314_full_circuit_w_senders/20180314_1':0,
'180316/20180316_fullcircuit_longpads_1':0,
'180320/180320_fullcircuit_longpads_1':0,
'180601/180601_1':-2.8889,
'180711/20180711_1':3.4375,
'180822_movie/20180822_1':0,
'181222_movie/20181222_1':0,
'190126_rhl_strain_6/20190126_2':0,
'190126_rhl_strain_6/20190126_4':0,
'190221_inducer_test/20190221_1':0,
'190222_pfb_test/20190222_1':0,
'190725_osc2/data':0,
'190726_i52_2/ladder_img_1':0,
'190727_pfbs/img_2':0,
'191018_rpa_senders_cin_pulse/img_1':90,
'191018_rpa_senders_cin_pulse/img_2':90,
'180316/20180316_fullcircuit_longpads_1':0,
'180316/20180316_fullcircuit_longpads_1am_1':0,
'180316/20180316_fullcircuit_longpads_2_3':0,
'20180319/20180319_fullcircuit_longpads_1':0,
'movies/20180420_AGAIN_1':5.5,
'20180509_three_fc_variants/20180509_1':0,
'20180517_i53_2_nosender/20180517_3':0,
'20181208_rhlpulsetry/20181208_1':0,
'20181208_rhlpulsetry/20181208_after_failure_1':0,
'20181208_rhlpulsetry/20181208_fewwer_points_1':0,
'movies/20181211_1':0,
'movies/20190108_1':0,
'movies/20190108_again_1':0,
'movies/20190110_both_pulsers_1':0,
'20190122_small_rhl_pads/20190122_1':0,
'20190122_small_rhl_pads/20190122_b_1':0}

# Functions
def debug_log(error, msg=None, traceback=None):
    with open('debug.log','a') as f:
        f.write(time.asctime()+"\n")
        f.write('{}\n'.format(error))
        if msg:
            f.write('{}\n'.format(msg))
        if traceback:
            f.write('{}\n'.format(traceback))
        f.write('\n\n')

def img_metadata_dict(img):
    '''
    Parameters:
        img : skimage.external.tifffileTiffFile object pointing to tiff (image or stack)
    Returns:
        dictionary of metatdata key value pairs as stored in the TIFF file
    Read metadata into dictionary from the TIFF metadata.
    This function uses string replacements to read dictionary key-value pairs.
    '''
    metadata_key = 50839
    metadata_str = img.tag[metadata_key].decode('utf_8').replace("\x00","")
    splits = re.split(",",metadata_str)
    metadata_keys = []
    metadata_vals = []
    for split_x in splits:
        hits = re.findall(r"(?<=\").*?(?=\")", split_x)
        if len(hits) == 3:
            key, _, val = [hit.replace("'","").strip() for hit in hits]
            metadata_keys.append(key)
            metadata_vals.append(val)
    return dict(zip(metadata_keys, metadata_vals))

def img_metadata_dict_full(img):
    '''Read metadata into dictionary from the TIFF metadata.

        After some cleanup, this function reads the metadata string directly as a dictionary definition.
    '''
    metadata_key = 50839
    metadata_str = img.tag[metadata_key].decode('utf-8').replace("\x00","")
    in_str = metadata_str[9:].replace('\n','').replace('\s*','').replace('null','None').replace('false','False').replace('true','True')
    out_dict = ast.literal_eval(in_str)
    return out_dict

def fn_metadata(fn):
    '''Wrapper for reading full metadata dictionary from a filename'''
    with Image.open(fn) as img:
        out = img_metadata_dict(img)
    return out

def fn_metadata_full(fn):
    '''Wrapper for reading full metadata dictionary from a filename'''
    with Image.open(fn) as img:
        out = img_metadata_dict_full(img)
    return out

def ctime(fname):
    '''Read creation time from the TIFF metadata in seconds from Epoch'''
    metadata = fn_metadata(fname)
    time_tuple = time.strptime(metadata['Time'][:-6],r"%Y-%m-%d %H:%M:%S")
    return time.mktime(time_tuple)

def ctime_parworker(fn_list, out_fn):
    n_fn = len(fn_list)
    ctime_vec = [ctime(xx) for xx in fn_list]
    index = pd.Index(data=fn_list, copy=True, name='fn')
    out_series = pd.Series(ctime_vec, index=index, name='time')
    out_series.to_csv(out_fn)

def ctime_parallel_wrapper(img_files):
    n_proc = 4
    sub_lists = [img_files[i::n_proc] for i in np.arange(n_proc)]
    out_fn_list = ['sub_time_{}.csv'.format(xx) for xx in np.arange(n_proc)]
    worker_inputs = zip(sub_lists, out_fn_list)
    with multiprocessing.Pool(n_proc) as pool:
        pool.starmap(ctime_parworker, worker_inputs)

def ctime_parallel_read_result(img_files):
    fn_list = [xx for xx in os.listdir() if 'sub_time' in xx]
    series = pd.concat([pd.read_csv(fn, index_col=0, header=None,names=['fn','time'],squeeze=True) for fn in fn_list])
#     series.index = [os.path.abspath(xx) for xx in series.index]
    return series.loc[img_files].values

def find_tiffs_in_dir(super_dir):
    pos_dirs = [os.path.join(super_dir, xx) for xx in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, xx))]
    img_files = [os.path.abspath(os.path.join(pos_dir, xx)) for pos_dir in pos_dirs for xx in os.listdir(pos_dir) if '.tif' in xx]
    return img_files

def get_timevec_par(img_files):
    ctime_parallel_wrapper(img_files)
    time_vec = ctime_parallel_read_result(img_files)
    time_vec = time_vec - time_vec.min()
    return time_vec

def get_timevec(img_files):
    time_vec = np.array([ctime(fn) for fn in img_files])
    # time_vec = time_vec - time_vec.min()
    return time_vec

def get_timedf(img_files, time_df_fn):
    n_files = len(img_files)
    time_df = pd.DataFrame(data={'keep':np.zeros(n_files),'time':np.zeros(n_files),'fn':img_files}).groupby('fn').agg(np.min)
    for img_fn in img_files:
        try:
            time_df.loc[img_fn, 'time'] = ctime(img_fn)
            time_df.loc[img_fn, 'keep'] = 1
        except (UnidentifiedImageError, KeyError, UnicodeDecodeError) as err:
            print(err)
            print('continuing get_timedf {}'.format(time_df_fn))
            with open('debug.log','a') as f:
                f.write(time.asctime()+"\n")
                f.write('error reading ctime in {}\ncontinuing get_timedf'.format(img_fn, time_df_fn))
    # Throw out bad image files
    time_df.drop(time_df.index[time_df.keep == 0], axis=0, inplace=True)
    time_df = time_df.reset_index().sort_values(by='time')
    print('printed {}'.format(time_df_fn))
    return time_df

def get_exp_summary_from_fn(fn):
    all_metadata = fn_metadata_full(fn)
    summ_dict = all_metadata['Summary']
    chan_ind_list = [int(xx) for xx in summ_dict['ChNames']]
    chan_names = [all_channel_names[xx] for xx in chan_ind_list]
    n_chan = len(chan_ind_list)
    im_width = summ_dict['Width']
    im_height = summ_dict['Height']
#    binning = all_metadata['Binning']
#    objective = summ_dict['Objective-State']
    return n_chan, chan_ind_list, chan_names, im_width, im_height

def make_re_patterns(super_dir):
    # Interpret pad, frame, channel from filenames
    posname_pattern = r"(?P<posname>[a-zA-Z\-0-9_]+)"
    filename_pattern = r"(?P<imgprefix>[a-zA-Z\-0-9]+)_(?P<frame>[0-9]*)_(?P<channel>[0-9]*)_(?P<zslice>[0-9]*).tif"
    full_filepath_pattern = os.path.join(super_dir, posname_pattern, filename_pattern)
    # "central" is sometimes prepended to the filepath. remove it, if present, and 
    # add a conditional group to the regex pattern
    full_filepath_pattern = full_filepath_pattern.replace(r"/central","")
    img_re_pattern = r"^(?:/central)?{}$".format(full_filepath_pattern)
    return posname_pattern, filename_pattern, full_filepath_pattern, img_re_pattern

def make_files_df(super_dir, overwrite=False):
    filepath_separates = os.path.abspath(super_dir).split(os.path.sep)
    expname = filepath_separates[-2]
    acqname = filepath_separates[-1]
    dict_key = os.path.join(expname, acqname)
    img_files = find_tiffs_in_dir(super_dir)
    n_files = len(img_files)
    if n_files < 1:
        print('{} no images found'.format(super_dir))
        print('write debug')
        with open('debug.log','a') as f:
            f.write(time.asctime()+'\n')
            f.write('{} no images found\n'.format(super_dir))
        return 1

    posname_pattern, filename_pattern, full_filepath_pattern, img_re_pattern = make_re_patterns(super_dir)
    re_matches = re.findall(img_re_pattern, '\n'.join(img_files), re.MULTILINE)
    n_matches = len(re_matches)
    if n_matches < 1:
        print('{} img full filename re failed with pattern {}'.format(super_dir, img_re_pattern))
        print('write debug')
        with open('debug.log','a') as f:
            f.write(time.asctime()+"\n")
            f.write('failed while making files_df\n')
            f.write('\n'.join(img_files))
            f.write('\n')
        return 1
    # Only matching files should be considered. Remake filenames from pattern matches
    img_str_tmpl = re.sub(r"\(\?.+?\)","{}", full_filepath_pattern)
    img_files = [img_str_tmpl.format(*xx) for xx in re_matches]
    # Get timedf 
    time_df_fn = os.path.abspath(os.path.join(super_dir, 'time_df.csv'))
    if not overwrite and os.path.isfile(time_df_fn):
        time_df = pd.read_csv(time_df_fn, index_col=None)
        time_vec, img_files = time_df[['time','fn']].values.T
    else:
        time_df = get_timedf(img_files, time_df_fn)
        time_df.to_csv(os.path.join(super_dir, 'time_df.csv'), index=False)
        time_vec, img_files = time_df[['time','fn']].values.T
    # get_timedf will skip files with malformed metadata.
    # keep only files with metadata
    n_files = len(img_files)
    re_matches = re.findall(img_re_pattern, '\n'.join(img_files), re.MULTILINE)
    rem_arr = np.array(re_matches,dtype=object)
    # rem_arr[rem_arr==''] = np.nan
    posname_vec, imgprefix_vec, frame_vec, channel_vec, zslice_vec = rem_arr.T

    # at this point, img_files contains all the files to be considered hereonout
    # so go ahead and make the dataframe
    columns = ['posname','imgprefix','frame','channel','zslice','fn','time','pad','padcol','padrow', 'pos']
    col_data = [posname_vec, imgprefix_vec, frame_vec, channel_vec, zslice_vec, img_files, time_vec,
               -np.ones(n_files),-np.ones(n_files),-np.ones(n_files),-np.ones(n_files)]
    dtypes = [str,str,np.int,np.int,np.int,str,np.int,np.int,np.int,np.int]
    col_data = [xx.astype(yy) for xx, yy in zip(col_data, dtypes)]
    files_df = pd.DataFrame(dict(zip(columns, col_data)))
    files_df.sort_values(by='time', inplace=True)

    # Extract pad information from posname_vec where you can
    # tag each posname with its index 
    array_pattern = r"fileind(?P<fileind>[0-9]+)-(?P<pad>[0-9]+)-Pos_(?P<padcol>[0-9]+)_(?P<padrow>[0-9]+)"
    posname_col = files_df['posname']
    tagged_posnames = ['fileind{}-{}'.format(indx, posname_col.loc[indx]) for indx in files_df.index]
    re_matches = re.findall(array_pattern, '\n'.join(tagged_posnames), re.MULTILINE)
    n_matches = len(re_matches)
    rem_arr = np.array(re_matches,dtype=object)
    if n_matches > 0:
      rem_arr[rem_arr==''] = np.nan
      colnames = ['pad','padcol','padrow']
      indx_vec, pad_vec, padcol_vec, padrow_vec = [xx.astype(np.int) for xx in rem_arr.T]
      for colname, colvec in zip(colnames, [pad_vec, padcol_vec, padrow_vec]):
        files_df.loc[indx_vec, colname] = colvec

    # Assign integer values to each position. If any are missed, throw exception
    gb_pos = files_df.groupby(['posname'])
    for i, inds in enumerate(gb_pos.groups.values()):
        files_df.loc[inds,'pos'] = i
    assert not np.any(files_df.pos.values == -1)
    res = apply_manual_pad_labeling(files_df, dict_key)
    if np.any(files_df['pad'].values<0):
        print(super_dir, 'not all pads assigned')
    # reassign integer values to each pad. If any are missed, throw exception
    gb_pad = files_df.groupby(['pad'])
    for i, inds in enumerate(gb_pad.groups.values()):
        files_df.loc[inds,'pad'] = i
    return files_df

def apply_manual_pad_labeling(files_df, key):
  if key in pad_metadata_dict:
      unique_positions = np.sort(np.unique(files_df.pos.values))
      pad_positions_tuples = np.array(pad_metadata_dict[key])
      gb_pos = files_df.groupby('pos')
      pad_ind = files_df['pad'].max()+1
      for positions_tuple in pad_positions_tuples:
          update_pos = np.array(positions_tuple)
          for pos in update_pos:
            inds = gb_pos.get_group(pos).index
            files_df.loc[inds,'pad'] = pad_ind
          pad_ind += 1
      return 0
  else:
      return 1

def make_positions_df(files_df):
    n_pads = len(np.unique(files_df['pad']))
    n_frames = np.max(files_df.frame)+1
    pos_df = files_df.loc[:,['pos','pad','fn']].groupby('pos').agg(np.min)
    n_pos = len(pos_df)
    if n_pos ==1:
        columns = ['x','y','label','pad', 'dist']
        n_cols = len(columns)
        cor_pos_df = pd.DataFrame(np.empty((n_pos,n_cols)), columns=columns, index=np.arange(n_pos))
        for pos in np.arange(n_pos):
            label = pos_df.loc[pos,'fn'].split('/')[-2]
            x, y = 0,0
            pad = pos_df.loc[pos,'pad']
            cor_pos_df.loc[pos,['x', 'y', 'label', 'pad']] = [x, y, label, pad]
        return cor_pos_df

    all_metadata = fn_metadata_full(files_df['fn'].values[0])
    if type(all_metadata) is int:
        if all_metadata == 1:
            print('could not read metadata from {}'.format(files_df['fn'].values[0]))
            with open('debug.log','a') as f:
                f.write(time.asctime()+"\n")
                f.write('could not read metadata from {} \n'.format(files_df['fn'].values[0]))
            return 1

    summ_dict = all_metadata['Summary']
    label_vec = [xx['Label'] for xx in summ_dict['InitialPositionList']]
    xy_vec = [xx['DeviceCoordinatesUm']['XYStage'] for xx in summ_dict['InitialPositionList']]
    label_xy_dict = dict(zip(label_vec, xy_vec))

    # Cor_pos_df is corrected position DF. In this notebook, correcting the position is not
    # necessary as there are only two imaging positions per pad. DF also includes inducer info
    # 0, 1, 2 for inducers blank, C, R; respectively
    columns = ['x','y','label','pad', 'dist']
    n_cols = len(columns)
    cor_pos_df = pd.DataFrame(np.empty((n_pos,n_cols)), columns=columns, index=np.arange(n_pos))
    for pos in np.arange(n_pos):
        label = pos_df.loc[pos,'fn'].split('/')[-2]
        x, y = label_xy_dict[label]
        pad = pos_df.loc[pos,'pad']
        cor_pos_df.loc[pos,['x', 'y', 'label', 'pad']] = [x, y, label, pad]
    return cor_pos_df

def plot_positions(cor_pos_df):
    plt.figure(figsize=(18,18))
    n_pads = len(cor_pos_df.groupby('pad'))
    colors = sns.color_palette('Set1', n_colors=n_pads)
    for p_i in cor_pos_df.index:
        point_color = colors[np.int(cor_pos_df.loc[p_i,"pad"])]
        plt.plot(np.float(cor_pos_df.x[p_i]),
                 -np.float(cor_pos_df.y[p_i]),
                 '.',
                 label=cor_pos_df['pad'][p_i],
                 ms=20,
                 c=point_color)
        plt.text(np.float(cor_pos_df.x[p_i]),
                 -np.float(cor_pos_df.y[p_i]),
                 '{}'.format(cor_pos_df.loc[p_i, 'label']),
                 fontsize=14,
                 rotation=15)
    handles = [Line2D([0],[0],color=colors[xx],marker='o',label=xx) for xx in np.arange(n_pads)]
    plt.legend(handles=handles)
    plt.gca().set_aspect('equal')
    return plt.gcf()

class WriteHelper():
  def _set_scale(self, scale):
    acq = self.acq
    pixel_size = acq.pixel_size * scale
    _, _, _, im_width, im_height = get_exp_summary_from_fn(acq.files_df.fn.values[0])
    pad_df = self.pad_df
    pos_lims = (pad_df[['x','y']]/pixel_size).astype(np.int)
    h, w = im_height//scale, im_width//scale
    self.one_arr = skimage.transform.rotate(np.ones((h,w)),self.rotation,resize=True)
    h, w = self.one_arr.shape
    pos_lims = (pad_df[['x','y']]/pixel_size).astype(np.int)
    xlims = np.array([pad_df.x.min(), pad_df.x.max()])/pixel_size + w*np.array([0,1])
    ylims = np.array([pad_df.y.min(), pad_df.y.max()])/pixel_size + h*np.array([0,1])
    rel_mins = np.concatenate([[xlims], [ylims]],axis=0).astype(np.int)
    pad_h = np.int(np.ceil(np.diff(ylims)))
    pad_w = np.int(np.ceil(np.diff(xlims)))

    # Scale such that figure fits in bounding box 20x20
    max_dim = np.max([pad_h, pad_w])
    fig_factor = self.fig_dim/max_dim
    fig_h, fig_w = np.array([pad_h, pad_w])*fig_factor

    count_arr = np.zeros((pad_h, pad_w),dtype=np.float32)
    pos_list = self.pad_df.index
    for pos in pos_list:
        x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
        count_arr[y0:y0+h,x0:x0+w] += self.one_arr
    self.count_arr,self.uncovered_arr,self.covered_arr = count_arr,count_arr<=0,count_arr>0
    self.count_arr[self.uncovered_arr] = 1
    self.h, self.w = h, w
    self.fig_h, self.fig_w = fig_h, fig_w
    self.pixel_size = pixel_size
    self.rel_mins,self.pos_list,self.pos_lims = rel_mins,pos_list,pos_lims
    self.pad_arr = np.zeros((pad_h, pad_w, self.acq.n_chan),dtype=np.float32)
    self.img_arr = np.zeros((pad_h, pad_w, 3),dtype=np.float32)
    self.norm_arr = np.zeros((pad_h, pad_w, 3),dtype=np.uint8)

  def __init__(self, acq, scale=8, pad_ind=None, bg_option='default', sigma=None, bg_sigma=None):
    tiff_dir = os.path.join('/central','scratchio','jparkin','tiffstacks',acq.expname,acq.acqname)
    if not os.path.exists(tiff_dir):
      os.makedirs(tiff_dir)
    self.tiff_dir = tiff_dir
    self.tiff_fn = os.path.join(tiff_dir, 'pad{}.tif'.format(pad_ind))
    self.fig_dim = 20
    self.sigma = sigma
    self.bg_sigma = bg_sigma
    self.bg_option = bg_option
    if pad_ind is None:
      pad_ind = 0
    self.pad_ind = pad_ind

    # Select bg option
    bg_options = ['default', 'init', 'pos_list', 'comb']
    bg_fns = [self.def_load_bg_img, self.load_init_bg_img, self.load_bg_poslist_img, self.load_comb_bg_img]
    bg_dict = dict(zip(bg_options, bg_fns))
    self.load_bg_img = bg_dict[bg_option]

    # Get movie metadata
    self.acq = acq
    uint_max = 65535

    # Setup plotting variables
    self.pad_df = acq.cor_pos_df.groupby('pad').get_group(pad_ind)
    self.chan_vec = acq.chan_ind_list
    self.pixel_size=acq.pixel_size
    self.scale,self.rotation = scale,acq.rotation
    self._set_scale(scale)

  def def_load_bg_img(self, frame_ind, pos, channel):
    return 0

  def load_init_bg_img(self, frame_ind, pos, channel):
    bg_im = self.acq.load_img(0, pos, channel, self.scale, self.bg_sigma)
    return bg_im

  def load_bg_poslist_img(self, frame_ind, pos, channel):
    pos_list = self.acq.bg_pos_list
    im_list = [ self.acq.load_img(frame_ind, bg_pos, channel, self.scale, self.bg_sigma) for bg_pos in pos_list ]
    #bg_im = np.median(np.array(im_list), axis=0)
    bg_im = np.mean(np.array(im_list), axis=0)
    return bg_im

  def load_comb_bg_img(self, frame_ind, pos, channel):
    if frame_ind > 0:
      im_list = [ fn(frame_ind, pos, channel) for fn in [self.load_init_bg_img, self.load_bg_poslist_img] ]
      #bg_im = np.median(np.array(im_list), axis=0)
      bg_im = np.mean(np.array(im_list), axis=0)
    else:
      bg_im = self.load_init_bg_img(frame_ind, pos, channel)
    return bg_im

  def get_bgsub_arr(self, frame, pos, channel):
    # frame_arr = np.zeros((h, w), dtype=np.float32)
    frame_arr = self.acq.load_img(frame, pos, channel, scale=self.scale, sigma=self.sigma)
    frame_arr -= self.load_bg_img(frame, pos, channel)
    return frame_arr

  def get_frame_arr(self, frame_ind):
    h, w = self.one_arr.shape
    self.pad_arr[:] = 0
    frame_arr = np.zeros((h, w), dtype=np.float32)
    chan_vec = self.chan_vec
    n_chan = len(chan_vec)
    chan_dict = dict(zip(chan_vec, np.arange(n_chan)))
    for chan_i, channel in enumerate(self.chan_vec):
      for pos in self.pos_list:
        x0, y0 = (self.pos_lims.loc[pos,:].values - self.rel_mins[:,0]).astype(np.int)
        yslice = slice(y0,y0+h)
        xslice = slice(x0,x0+w)
        frame_arr[:] = self.get_bgsub_arr(frame_ind, pos, channel)
        self.pad_arr[yslice,xslice,chan_i] += frame_arr / self.count_arr[yslice, xslice]
    # Correct for cross-channel interference
    inter_tuples = [(3,2,0.0585)]
    inter_df = pd.DataFrame(inter_tuples, columns=['source_chan','dest_chan','rate'])
    gb_source_chan = inter_df.groupby('source_chan')
    for source_chan in inter_df.source_chan.values:
      sub_inter_df = gb_source_chan.get_group(source_chan)
      dest_chan, rate = sub_inter_df[['dest_chan','rate']].values[0]
      source_chan_i, dest_chan_i = (chan_dict[source_chan], chan_dict[dest_chan])
      self.pad_arr[:,:,dest_chan_i] -= rate*self.pad_arr[:,:,source_chan_i]

  def _prep_img_arr(self):
    pad_h, pad_w = self.pad_arr.shape[:2]
    self.img_arr[:] = 0
    # Use contrast-stretching approach, where the bounds are the middle 90 percentile
    norm_vec = []
    for chan_ind in np.arange(self.acq.n_chan):
      frame_vals = self.pad_arr[:,:,chan_ind][self.covered_arr].flatten()
      frame_vals = np.sort(np.unique(frame_vals))
      n_vals = len(frame_vals)
      ind_min, ind_max = np.array([0.1*n_vals, 1*n_vals-1],dtype=np.int)
      vmin, vmax = frame_vals[ind_min], frame_vals[ind_max]
      vmin = np.max([20,vmin])
      vmax = np.max([vmax+vmin,vmin*2])
      norm_fn = mpl_colors.Normalize(vmin, vmax, clip=True)
      norm_vec.append(norm_fn)
    for chan_ind, chan_slot in enumerate(self.chan_vec):
      norm = norm_vec[chan_ind]
      chan_arr = self.pad_arr[:,:,chan_ind:chan_ind+1]
      color_vec = mpl_colors.to_rgb(mpl_named_colors[chan_slot])
      self.img_arr += np.concatenate([norm(chan_arr)*color_val for color_val in color_vec],axis=2)

  def _prep_norm_arr(self):
    norm_fn = mpl_colors.Normalize(0, 1, clip=True)
    self.norm_arr[:,:,:] = (255*norm_fn(self.img_arr)).astype(np.uint8)

  def animate(self, frame_ind):
    pad_h, pad_w = self.pad_arr.shape[:2]
    self.img_arr[:] = 0
    self.get_frame_arr(frame_ind)
    self._prep_img_arr()
    self._prep_norm_arr()
    self.im.set_array(self.norm_arr)

  def setup_plot(self):
    fig, ax = plt.subplots(1, 1, figsize=(self.fig_w, self.fig_h+1.2))
    self.im = ax.imshow(np.zeros((self.h, self.w), dtype=np.uint8), animated=True, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(self.out_fn)
    fig.tight_layout()
    return fig, ax

  def save_anim(self, out_fn, writer='ffmpeg', anim_fn=None):
    if anim_fn is None:
      anim_fn = self.animate
    self.out_fn = out_fn
    fig, ax = self.setup_plot()
    anim = anm.FuncAnimation(fig, anim_fn, interval=100, frames=self.frame_vec)
    plt.close('all')
    anim.save(out_fn, dpi=80, fps=3, writer=writer)

  def load_movie(self):
    n_frames = len(self.acq.frame_vec)
    im_h, im_w, n_chan = self.pad_arr.shape
    arr = np.zeros((n_frames, n_chan, im_h, im_w))
    for frame in self.acq.frame_vec:
      self.get_frame_arr(frame)
      for ci in np.arange(n_chan):
        arr[frame,ci,:,:,] = self.pad_arr[:,:,ci].copy()
    return arr

  def save_tiffstack(self, tiff_fn):
    keys = ['scale','bg_option','sigma','bg_sigma']
    values = self.scale, self.bg_option, self.sigma, self.bg_sigma
    metadata = dict(zip(keys, values))
    tiff_dir = os.path.dirname(tiff_fn)
    assert os.path.isdir(tiff_dir)
    n_chan = self.acq.n_chan
    with skimage.external.tifffile.TiffWriter(tiff_fn, imagej=True) as tif:
      for frame in self.acq.frame_vec:
        self.get_frame_arr(frame)
        for ci in np.arange(n_chan):
          arr = self.pad_arr[:,:,ci]
          tif.save(arr, metadata=metadata)
    return tiff_fn

  def _get_tiff_metadata(self, tiff_fn):
    tfile = skimage.external.tifffile.TiffFile(tiff_fn)
    md_lines = tfile.info().split('\n')
    imj_flag = False
    imj_lines = []
    for line in md_lines:
      if imj_flag:
        key, val = line.split(':')
        imj_lines.append((key[2:], val[1:]))
      if '* ImageJ' in line:
        imj_flag = True
    md_dict = dict(imj_lines)
    return md_dict

  def _check_tiff_metadata(self, tiff_fn):
    md_dict = self._get_tiff_metadata(tiff_fn)
    md_keys = ['scale','sigma','bg_sigma','bg_option']
    md_vals = ['{}'.format(getattr(self, xx)) for xx in md_keys]
    true_dict = dict(zip(md_keys,md_vals))
    if not np.all([xx in md_dict for xx in md_keys]):
      return False
    check_vec = [md_dict[key]==true_dict[key] for key in md_keys]
    return np.all(check_vec)

  def load_tiffstack(self, fn, memmap=False):
    arr = skimage.external.tifffile.imread(fn, memmap=memmap)
    nh, nw, nc = self.pad_arr.shape
    nf = len(self.acq.frame_vec)
    arr_shape = (nf, nc, nh, nw)
    arr = arr.reshape(arr_shape)
    return arr

  def save_frame(self, out_fn, frame_ind):
    self.out_fn = out_fn
    fig, ax = self.setup_plot()
    self.animate(frame_ind)
    fig.savefig(out_fn)
    plt.close('all')

class Acquisition():
    '''
    '''
    def __init__(self, in_str, overwrite_files_df=False, overwrite_cor_pos_df=False, overwrite_time_df=False):
        self.success = False
        if os.path.isdir(in_str):
            res = self._init_dir(in_str, overwrite_files_df, overwrite_cor_pos_df, overwrite_time_df)
        elif os.path.isfile(in_str):
            res = self._init_file(in_str)
        else:
            raise(ValueError("Input string does not specify a directory or file"))
        if res == 0:
          self._set_metadata()
          self.success = True
        elif res == 1:
            with open('debug.log','a') as f:
                f.write(time.asctime()+"\n")
                f.write('end {} debug \n'.format(in_str))

    def _init_dir(self, super_dir, overwrite_files_df=False, overwrite_cor_pos_df=False, overwrite_time_df=False):
        self.super_dir = os.path.abspath(super_dir)
        filepath_separates = os.path.abspath(self.super_dir).split(os.path.sep)
        self.expname = filepath_separates[-2]
        self.acqname = filepath_separates[-1]
        files_df_fn = os.path.abspath(os.path.join(super_dir, 'files_df.csv'))
        cor_pos_df_fn = os.path.abspath(os.path.join(super_dir, 'cor_pos_df.csv'))
        self.files_df_fn = files_df_fn
        self.cor_pos_df_fn = cor_pos_df_fn
        if not overwrite_files_df and os.path.isfile(files_df_fn):
            self.files_df = pd.read_csv(files_df_fn, index_col=None)
        else:
            self.files_df = make_files_df(super_dir, overwrite_time_df)
            if type(self.files_df) is not pd.DataFrame:
                if self.files_df == 1:
                    print('error making filesdf, aborting dir {}'.format(super_dir))
                    return 1
            self.files_df.to_csv(files_df_fn, index=False)
            print('finished printing {}'.format(files_df_fn))
        if not overwrite_cor_pos_df and os.path.isfile(cor_pos_df_fn):
            self.cor_pos_df = pd.read_csv(cor_pos_df_fn, index_col=None)
        else:
            self.cor_pos_df = make_positions_df(self.files_df)
            if type(self.cor_pos_df) is not pd.DataFrame:
                if self.cor_pos_df == 1:
                    print('error making corposdf, aborting dir {}'.format(super_dir))
                    return 1
            self.cor_pos_df.to_csv(cor_pos_df_fn, index=False)
            print('finished printing {}'.format(cor_pos_df_fn))
        for fn in [files_df_fn, cor_pos_df_fn]:
            assert os.path.isfile(fn)
        return 0

    def _set_metadata(self):
        dict_key = os.path.join(self.expname,self.acqname)
        if hasattr(self, 'files_df'):
            self.frame_vec = np.sort(np.unique(self.files_df.frame.values))
            if dict_key in pad_mag_dict.keys():
                self.magoption16 = pad_mag_dict[dict_key]
            else:
                self.magoption16 = False
            if dict_key in pad_bg_pos_lists.keys():
                self.bg_pos_list = pad_bg_pos_lists[dict_key]
            else:
                self.bg_pos_list = None
            fn = self.files_df.fn.values[0]
            all_metadata = fn_metadata_full(fn)
            summ_dict = all_metadata['Summary']
            self.chan_ind_list = [int(xx) for xx in summ_dict['ChNames']]
            self.chan_names = [all_channel_names[xx] for xx in self.chan_ind_list]
            self.n_chan = len(self.chan_ind_list)
            self.im_width = summ_dict['Width']
            self.im_height = summ_dict['Height']
            self.binning = all_metadata['Binning']
            if dict_key in pad_obj_dict.keys():
                self.objective = pad_obj_dict[dict_key]
            else:
              # self.objective = all_metadata['Objectives-State']
                self.objective = '5'
            self.pixel_size = pixel_size_table.loc[self.objective, self.binning]
            if dict_key in pad_rotation_dict:
              self.rotation = pad_rotation_dict[dict_key]
            else:
              thresh_tuple = time.strptime(r"2019-07-20 12:00:00",r"%Y-%m-%d %H:%M:%S")
              thresh_time = time.mktime(thresh_tuple)
              if self.files_df.time.max() < thresh_time:
                  self.rotation = 0
              else:
                  self.rotation = 180
            if self.magoption16:
                self.pixel_size = self.pixel_size/1.6
            gb = self.files_df[['frame','pad','time']].groupby(['frame','pad'])
            time_df = gb.agg(np.mean)
            time_df['time'] = time_df['time'] - time_df.time.min()
            time_df = time_df.sort_index()
            self.time_df = time_df
        else:
            print('initialize with files_df')

    def _init_file(self, files_df_fn):
        self.super_dir = os.path.dirname(os.path.abspath(files_df_fn))
        cor_pos_df_fn = os.path.abspath(os.path.join(self.super_dir, 'cor_pos_df.csv'))
        filepath_separates = os.path.abspath(self.super_dir).split(os.path.sep)
        self.expname = filepath_separates[-2]
        self.acqname = filepath_separates[-1]
        self.files_df = pd.read_csv(files_df_fn, index_col=None)
        self.cor_pos_df = make_positions_df(self.files_df)
        self.files_df_fn = files_df_fn
        self.cor_pos_df_fn = cor_pos_df_fn
        return 0

    def plot_positions(self, out_fn=None):
        if out_fn is None:
            out_fn = '{}_{}_cor_pos.png'.format(self.expname, self.acqname)
            out_fn = os.path.abspath(os.path.join('.','pngs', out_fn))
        print(out_fn)
        fig = plot_positions(self.cor_pos_df)
        plt.savefig(out_fn)
        plt.close('all')

    def load_img(self, frame, pos, channel, scale=1, sigma=None):
        pad = self.cor_pos_df.loc[pos,'pad']
        frame_bool = self.files_df.frame==frame
        pos_bool = self.files_df.pos==pos
        chan_bool = self.files_df.channel == channel
        indx_bool = (frame_bool)&(pos_bool)&(chan_bool)
        h, w = self.im_height, self.im_width
        if sum(indx_bool) < 1:
          error_msg = 'Image not found when printing {}: pad:{} frame:{} pos:{} channel:{}'
          error_msg = error_msg.format(self.expname, pad, frame, pos, channel)
          print(error_msg)
          img = np.zeros((h,w),dtype=np.float32)
          img = skimage.transform.downscale_local_mean(img, (scale, scale))
          return skimage.transform.rotate(img, self.rotation, resize=True)
        fn = self.files_df.loc[indx_bool,'fn'].values[0]
        try:
          img = skimage.io.imread(fn).astype(np.float32)
        except Exception as error:
          print('import img error {}'.format(fn))
          raise error
        if not (sigma is None):
          img = skimage.filters.gaussian(img, sigma, preserve_range=True)
        img = skimage.transform.downscale_local_mean(img, (scale, scale))
        return skimage.transform.rotate(img, self.rotation, resize=True)

    def _prep_writehelper(self, pad_ind, scale):
        args = pad_ind, self.files_df, self.cor_pos_df, self.pixel_size, scale, self.rotation

    def write_movie(self, out_fn, pad_ind, scale=8, skip=1, bg_option='default'):
        #out_fn = '{}_{}_pad_{}.gif'.format(self.expname, self.acqname, pad_ind)
        # out_fn = os.path.join(self.super_dir, out_fn)
        #out_fn = os.path.abspath(os.path.join('.','anims', out_fn))
        args = acq, scale, pad_ind, bg_option
        try:
            self.writer_obj = WriteHelper(*args)
            self.writer_obj.save_anim(out_fn, 'pillow')
        except FileNotFoundError as err:
            with open('bad_files.txt','a') as f:
                f.write(self.super_dir+"\n")
                print(err)

    def write_all_pad_gifs(self, out_fn, scale=8, skip=1, bg_option='default'):
        out_tmpl = '{}_{}_pad_{}.gif'
        out_tmpl = os.path.abspath(os.path.join('.','anims', out_tmpl))
        pad_inds = np.unique(self.files_df['pad'].values)
        for pad_ind in pad_inds:
            out_fn = out_tmpl.format(self.expname, self.acqname, pad_ind)
            self.write_movie(out_fn, pad_ind, scale, skip, bg_option)

    def write_frame(self, frame_ind, pad_ind, scale=8, bg_option='default'):
        out_fn = '{}_{}_pad_{}_frame_{}.png'.format(self.expname, self.acqname, pad_ind, frame_ind)
        out_fn = os.path.abspath(os.path.join('.','pngs', out_fn))
        print(out_fn)
        args = acq, scale, pad_ind, bg_option
        try:
          if self.bg_pos_list is None and bg_option!='default':
            bg_option = 'default'
          self.writer_obj = WriteHelper(*args)
          self.writer_obj.save_frame(out_fn, frame_ind)
        except FileNotFoundError as err:
            with open('bad_files.txt','a') as f:
                f.write(self.super_dir+" failed write_frame\n")
                print(err)

    def write_all_pad_frame(self, frame_ind, scale=8, bg_option='default'):
        pad_inds = np.unique(self.files_df['pad'].values)
        for pad_ind in pad_inds:
            self.write_frame(frame_ind, pad_ind, scale, bg_option='default')

    def write_all_bg(self, frame_ind, pad_ind):
        bpl = self.bg_pos_list
        out_fn = 'bg_frame_{}_chan_{}.tif'.format(self.expname, self.acqname, frame_ind, chan_ind)
        out_fn = os.path.abspath(os.path.join('super_dir','worker_outputs', out_fn))
        # print(out_fn)
        gb_fcp = self.files_df.groupby(['frame','channel','pos'])
        for frame_ind in self.frame_vec:
          for chan_ind in self.chan_ind_list:
            #bg_arr = np.empty((len(bpl), type=np.uint16)
            for i_pos, pos in enumerate(bpl):
              bg_arr[i_pos,:,:] = skimage.io.imread(gb_fcp.get_group((frame_ind, chan_ind, pos)))
            bg_img = np.median(bg_arr,axis=0)
            skimge.io.imsave(out_fn.format(frame_ind, chan_ind), bg_img)

        args = frame_ind, out_fn, pad_ind, self.files_df, self.cor_pos_df, self.pixel_size, scale, self.rotation
        try:
            write_frame_no_bg(*args)
        except FileNotFoundError as err:
            with open('bad_files.txt','a') as f:
                f.write(self.super_dir+"\n")
                print(err)

    def write_tiff_stack(self):
      tiff_tmpl = os.path.join('/central','scratchio','jparkin','{}', '{}.tif')

class ProcessPad():
  '''
  Create dataframe of background-subtracted experimental data from one pad.

  Processor class uses WriteHelper to grab bg-subtracted frames
  '''
  def __init__(self, acq, pad_ind, scale):
    self.acq = acq
    self.pad_ind = pad_ind
    self.scale = scale
    self.pad_helper = WriteHelper(acq, scale, pad_ind, bg_option='comb')
    columns = ['frame', 'x', 'y', 'pad', 'fluor', 'channel','scale']
    self.out_df = pd.DataFrame(columns=columns)
    ny, nx, nc = self.pad_helper.pad_arr.shape
    self.x_arr = np.tile(np.arange(nx).reshape((1,nx)),(ny,1))
    self.y_arr = np.tile(np.arange(ny).reshape((ny,1)),(1,nx))
    thresh_mins = [200, 200, 1e3]
    self.thresh_dict = dict(zip([2,3,5],thresh_mins))

  def process_frame(self, frame):
    acq, pad_helper = self.acq, self.pad_helper
    columns = ['frame', 'x', 'y', 'pad', 'fluor', 'channel', 'scale', 'thresh']
    out_df = pd.DataFrame(columns=columns, dtype=np.float)
    pad_helper.get_frame_arr(frame)
    pad_arr = pad_helper.pad_arr
    for c_i in np.arange(acq.n_chan):
        chan_arr = pad_arr[:,:,c_i]
        thresh = skimage.filters.threshold_li(chan_arr)
        thresh_min = self.thresh_dict[acq.chan_ind_list[c_i]]
        thresh = np.max([thresh_min, thresh])
        thresh_arr = chan_arr > thresh
        thresh_arr = skimage.morphology.remove_small_objects(thresh_arr, 50)
        n_thresh = np.sum(thresh_arr)
        if n_thresh > 0 :
            ones_vec = np.ones(n_thresh)
            x_vec = self.x_arr[thresh_arr].flatten()
            y_vec = self.y_arr[thresh_arr].flatten()
            fluor_vec = chan_arr[thresh_arr].flatten()
            frame_vec = frame*ones_vec
            chan_vec = acq.chan_ind_list[c_i]*ones_vec
            pad_vec = self.pad_ind*ones_vec
            scale_vec = self.scale*ones_vec
            thresh_vec = thresh*ones_vec
            data_cols = [frame_vec, x_vec, y_vec, pad_vec, fluor_vec, chan_vec, scale_vec, thresh_vec]
            update_df = pd.DataFrame(dict(zip(columns, data_cols)))
            out_df = pd.concat([out_df, update_df], ignore_index=True)
    return out_df
    #self.out_df = out_df

  def begin(self):
    columns = ['frame', 'x', 'y', 'pad', 'fluor', 'channel', 'scale', 'thresh']
    out_df = pd.DataFrame(columns=columns, dtype=np.float)
    # Check that csvs directory exists
    out_dir = os.path.join(self.acq.super_dir,"csvs")
    if not os.path.isdir(out_dir):
      os.mkdir(out_dir)
    out_tmpl = os.path.join(out_dir,"pad{}.csv")
    for frame in self.acq.frame_vec:
      in_df = self.process_frame(frame)
      out_df = pd.concat([in_df, out_df], axis=0, ignore_index=True)
    n_rows = len(out_df)
    acqname = "_".join(self.acq.super_dir.split(os.path.sep)[-2:])
    acqname_vec = np.repeat(acqname, n_rows)
    acqname_col = pd.DataFrame(acqname_vec, index=out_df.index, columns=['acqname'])
    out_df = pd.concat([out_df, acqname_col], axis=1)
    out_df.to_csv(out_tmpl.format(self.pad_ind), index=False)

class ComparisonAnimator():
    def _setup_imshow(self, ax):
        # setup blank imshow axis with no ticks
        norm_arr = self.processor.pad_helper.norm_arr
        handle = ax.imshow(norm_arr, animated=True, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        return handle 
    
    def _animate_movie(self, frame):
        wh = self.processor.pad_helper
        wh.get_frame_arr(frame)
        wh._prep_img_arr()
        wh._prep_norm_arr()
        self.ims[0].set_data(wh.norm_arr)
        self.axs[1].set_title(frame)
        
    def _prep_thresh_chan(self, im_dist, thresh_arr, c_i):
        processor, acq, wh = self.processor, self.processor.acq, self.processor.pad_helper
        if np.any(thresh_arr):
            m_dist = im_dist[thresh_arr].max()*np.ones_like(thresh_arr)
            dist_band = np.isclose(im_dist,m_dist,atol=3)
            wh.pad_arr[:,:,c_i] = (thresh_arr+1.5*dist_band)
        else:
            wh.pad_arr[:,:,c_i] = 0

    def _finish_thresh(self, frame):
        processor, acq, wh = self.processor, self.processor.acq, self.processor.pad_helper
        wh.pad_arr *= 4e4
        wh.pad_arr += np.random.random(wh.pad_arr.shape)
        wh._prep_img_arr()
        wh._prep_norm_arr()
        self.ims[1].set_data(wh.norm_arr)    
        
    def _animate_thresh(self, frame):
        processor, acq, wh = self.processor, self.processor.acq, self.processor.pad_helper
        im_dist, yfp_thresh, _ = processor._yfp_dist(frame)
        wh.pad_arr[:,:,0] = im_dist==0
        for chan in [1,2]:
            thresh_arr, chan_arr, thresh = processor._chan_thresh(frame, chan)
            if np.any(thresh_arr):
                m_dist = im_dist[thresh_arr].max()*np.ones_like(thresh_arr)
                dist_band = np.isclose(im_dist,m_dist,atol=3)
                wh.pad_arr[:,:,chan] = (thresh_arr+1.5*dist_band)
            else:
                wh.pad_arr[:,:,chan] = 0
        wh.pad_arr *= 4e4
        wh.pad_arr += np.random.random(wh.pad_arr.shape)
        wh._prep_img_arr()
        wh._prep_norm_arr()
        self.ims[1].set_data(wh.norm_arr)    
        
    def _animate_thresh_from_df(self, frame):
        processor, acq, wh = self.processor, self.processor.acq, self.processor.pad_helper
        im_dist, yfp_thresh, _ = processor._yfp_dist(frame)
        df = pd.read_csv(self.tmpl.format(self.processor.pad_ind, frame))
        wh.pad_arr[:] = 0
        x_vec, y_vec = df.loc[:,['x','y']].values.astype(np.int).T
        c_vec = (df.channel.values.astype(np.int)-1)//2
        wh.pad_arr[y_vec, x_vec, c_vec] = 1
        wh.pad_arr *= 4e4
        wh.pad_arr += np.random.random(wh.pad_arr.shape)
        wh._prep_img_arr()
        wh._prep_norm_arr()
        self.ims[1].set_data(wh.norm_arr)
        
    def _setup_figure(self):
        processor = self.processor
        oldfig = getattr(self, 'fig', None)
        if not (oldfig is None):
            plt.close(oldfig)
        fig, axs = plt.subplots(1,2,figsize=(10,7))
        self.fig = fig
        self.axs = axs
        self.ims = [self._setup_imshow(ax) for ax in axs]
        exp_name = '/'.join([processor.acq.expname, processor.acq.acqname])
        title = "{}\nPad:{}".format(exp_name, processor.pad_ind)
        axs[0].set_title(title)
        axs[0].set_xlabel('Fluorescence')
        axs[1].set_xlabel('Above threshold')
        
    def __init__(self, processor, step=1):
        self.step = step
        wh = processor.pad_helper
        wh.load_bg_img = wh.load_bg_poslist_img
        self.processor = processor
        acq = processor.acq
        self.tmpl = os.path.join(acq.super_dir,'csvs','pad{}_frame{}.csv')
        self._setup_figure()
        
    def animate(self, frame):
        self._animate_movie(frame)
        self._animate_thresh(frame)
        self.axs[1].set_title(frame)
        
    def get_animation(self):
        fig = self.fig
        func = self.animate
        frames = self.processor.acq.frame_vec[::self.step]
        anim = anm.FuncAnimation(fig, func, interval=70, frames=frames)
        return anim
    
class ProcessUnorderedDiff():
  '''
  Create dataframe of background-subtracted experimental data from one pad.
  Assumes yfp, rfp, cfp channels in that order

  Processor class uses WriteHelper to grab bg-subtracted frames
  '''
  def __init__(self, acq, pad_ind, scale, sigma=None, bg_sigma=None, 
      print_img=False, bg_option='default',
      diff=False, overwrite=False):
    # Setup attributes
    self.acq = acq
    self.pad_ind = pad_ind
    self.scale = scale
    self.diff = diff
    if sigma is None:
      self.sigma = 4/scale
    else:
      self.sigma=sigma
    if bg_sigma is None:
      self.bg_sigma = 64/scale
    else:
      self.bg_sigma = bg_sigma
    self.bg_option = bg_option
    self.minsize = 50/(scale**2)
    self.pad_helper = WriteHelper(acq, scale, pad_ind, bg_option=bg_option, 
        sigma=sigma, bg_sigma=bg_sigma)
    ny, nx, nc = self.pad_helper.pad_arr.shape
    self.x_arr = np.tile(np.arange(nx).reshape((1,nx)),(ny,1))
    self.y_arr = np.tile(np.arange(ny).reshape((ny,1)),(1,nx))
    tiff_dir = os.path.join('/central','scratchio','jparkin','tiffstacks',
        acq.expname,acq.acqname)
    if not os.path.exists(tiff_dir):
      os.makedirs(tiff_dir)
    self.tiff_dir = tiff_dir
    self._load_arrs(overwrite=overwrite)
    self._setup_thresh()

    # Print img setup
    self.print_img = print_img
    if print_img:
      self._setup_printer()

  def _setup_printer(self):
      self.printer = ComparisonAnimator(self)
      out_tmpl = 'frame{:02d}_pad{:02d}.png'
      if self.diff:
        out_tmpl = os.path.abspath(os.path.join('.','pngs','diff_thresh_progress', out_tmpl))
      else:
        out_tmpl = os.path.abspath(os.path.join('.','pngs','thresh_progress', out_tmpl))
      self.png_tmpl = out_tmpl

  def _save_tiffstack(self, tiff_fn, arr):
    keys = ['scale','bg_option','sigma','bg_sigma']
    values = self.scale, self.bg_option, self.sigma, self.bg_sigma
    metadata = dict(zip(keys, values))
    tiff_dir = os.path.dirname(tiff_fn)
    assert os.path.isdir(tiff_dir)
    skimage.external.tifffile.imsave(tiff_fn, arr, metadata=metadata, imagej=True)
    return tiff_fn

  def load_tiffstack(self, fn, memmap=False):
    arr = skimage.external.tifffile.imread(fn, memmap=memmap)
    return arr

  def _load_arrs(self, overwrite=False):
    acq = self.acq
    # make writehelper and setup arr
    n_frames = len(acq.frame_vec)
    im_h, im_w, n_chan = self.pad_helper.pad_arr.shape
    arr = np.zeros((n_frames, n_chan, im_h, im_w))
    scratch_dir = self.tiff_dir
    fn_stem = '{}pad{}.tif'
    scr_tmpl = os.path.join(scratch_dir, fn_stem)
    prefixes = ['', 'filt_', 'diff_']
    scratch_tiffs = [scr_tmpl.format(xx, self.pad_ind) for xx in prefixes]
    # Load movie arr
    tiff_fn = scratch_tiffs[0]
    load_flag = False
    if os.path.exists(tiff_fn):
      # Check metadata
      load_flag = self.pad_helper._check_tiff_metadata(tiff_fn)
    if load_flag:
      arr = self.pad_helper.load_tiffstack(tiff_fn)
    else:
      arr = self.pad_helper.load_movie()
    self.arr = arr

    # Load filt movie arr
    sg_filter = lambda arr : signal.savgol_filter(arr, 9, 1, axis=0)
    self.sg_filter = sg_filter
    tiff_fn = scratch_tiffs[1]
    load_flag = False
    if os.path.exists(tiff_fn):
      # Check metadata
      load_flag = self.pad_helper._check_tiff_metadata(tiff_fn)
    if load_flag and not overwrite:
      self.filt_arr = self.load_tiffstack(tiff_fn)
    else:
      self.filt_arr = sg_filter(arr)
      self._save_tiffstack(tiff_fn, self.filt_arr)

    # Load diff movie arr
    self.sg_filter = sg_filter
    tiff_fn = scratch_tiffs[2]
    load_flag = False
    if os.path.exists(tiff_fn):
      # Check metadata
      load_flag = self.pad_helper._check_tiff_metadata(tiff_fn)
    if load_flag and not overwrite:
      self.diff_arr = self.load_tiffstack(tiff_fn)
    else:
      self.diff_arr = sg_filter(np.diff(self.filt_arr, axis=0))
      self._save_tiffstack(tiff_fn, self.diff_arr)

    self.arr_dict = dict(zip(['arr','filt','diff'],[self.arr, self.filt_arr, self.diff_arr]))
    self.columns = ['frame', 'x', 'y', 'pad', 'fluor', 'channel', 
                    'scale', 'thresh', 'dist', 'rad']

  def _setup_thresh(self, overwrite=False):
    thresh_df_fn = os.path.join(self.acq.super_dir,"csvs",'empirical_thresholds.csv')
    if overwrite or not os.path.exists(thresh_df_fn):
      self.thresh_df = self._make_thresh_df(self.acq.bg_pos_list)
      self.thresh_df.to_csv(thresh_df_fn)
    else:
      self.thresh_df = pd.read_csv(thresh_df_fn, index_col=['type','ci','frame'])

    # Global thresh minds
    chan_inds = [2,3,5]
    self.thresh_min_dict = dict(zip([2,3,5],[1e3,5e2,4e3]))
    self.thresh_min_diff_dict = dict(zip([2,3,5],[40,30,170]))

  def _load_pad_arr(self, pad_ind):
    pad_helper = WriteHelper(self.acq,
                             scale=self.scale,
                             sigma=self.sigma,
                             bg_sigma=self.bg_sigma,
                             pad_ind=pad_ind,
                             bg_option=self.bg_option)
    if pad_helper._check_tiff_metadata(pad_helper.tiff_fn):
      return pad_helper.load_tiffstack(pad_helper.tiff_fn)
    else:
      return pad_helper.load_movie()

  def _make_thresh_df(self, bg_pads):
    # setup array shapes and load images
    im_h, im_w, n_chan = self.pad_helper.pad_arr.shape
    n_frames = len(self.acq.frame_vec)
    c_vec = np.arange(n_chan)
    stack_shape = (n_frames, n_chan, im_h, im_w, 1)
    img_list = [self._load_pad_arr(pad).reshape(stack_shape) for pad in bg_pads]
    # Make arrays
    bg_arrs = np.concatenate(img_list, axis=-1)
    filt_arrs = self.sg_filter(bg_arrs)
    diff_arrs = self.sg_filter(np.diff(filt_arrs, axis=0))
    thresh_df = pd.DataFrame(dtype=np.float)
    arr_labels = ['arr', 'filt', 'diff']
    bounds = [0.05,0.5,0.95]
    thresh_factor = [3,2][np.int(self.diff)]
    for label, arr in zip(arr_labels, [bg_arrs, filt_arrs, diff_arrs]):
      f_vec = np.arange(arr.shape[0])
      bound_arrs = np.quantile(arr, q=bounds, axis=(2,3,4))
      for ci in c_vec:
        bound_vecs = bound_arrs[:,:,ci]
        thresh_vec = thresh_factor*(bound_vecs[-1,:] - bound_vecs[0,:]) + bound_vecs[1,:]
        ind_tup = [(label, ci, frame) for frame in f_vec]
        multiindex = pd.MultiIndex.from_tuples(ind_tup, names=['type','ci','frame'])
        thresh_row = pd.DataFrame(index=multiindex, data=thresh_vec, columns=['threshold'])
        thresh_df = pd.concat([thresh_df, thresh_row], axis=0)
    return thresh_df

  def _chan_thresh(self, frame, c_i, arr_type='arr'):
    chan_arr = self.arr_dict[arr_type][frame, c_i,:,:]
    thresh_min = self.thresh_df.loc[(arr_type,c_i,frame),'threshold']
    initial_guess = np.quantile(chan_arr, 0.97)
    thresh = skimage.filters.threshold_li(chan_arr, tolerance=5, initial_guess=initial_guess)
    thresh = np.max((thresh, thresh_min))
    thresh_arr = chan_arr > thresh
    thresh_arr = skimage.morphology.remove_small_objects(thresh_arr,self.minsize)
    return thresh_arr, chan_arr, thresh

  def _yfp_dist(self, frame):
    ny, nx, nc = self.pad_helper.pad_arr.shape
    chan, c_i = 2, 0
    thresh_arr, chan_arr, thresh = self._chan_thresh(frame, c_i, arr_type='arr')
    if np.any(thresh_arr):
      d_coeff = self.acq.pixel_size * self.scale
      lab_arr, n_senders = skimage.morphology.label(thresh_arr, return_num=True)
      sen_labs = np.arange(n_senders)+1
      dist_arrs = [d_coeff*ndi.morphology.distance_transform_bf(lab_arr!=xx) for xx in sen_labs]
      rad_vec = [d_coeff*np.sqrt(np.sum(lab_arr==xx)/np.pi) for xx in sen_labs]
      im_dist = d_coeff*ndi.morphology.distance_transform_bf(False==thresh_arr)
      mask_arrs = [(im_dist == dist_arr) for dist_arr in dist_arrs]
      sum_masks = np.sum(mask_arrs,axis=0)
      rad_arrs = [rad*mask_arr for rad, mask_arr in zip(rad_vec, mask_arrs)]
      rad_arr = np.sum(rad_arrs,axis=0)/sum_masks
    else:
      im_dist = np.sqrt(np.power(self.x_arr-nx//2,2) + np.power(self.y_arr-ny//2,2))
      rad_arr = np.ones_like(im_dist)
    return im_dist, thresh, rad_arr

  def _chan_diff_thresh(self, frame, c_i):
    return self._chan_thresh(frame, c_i, 'diff')

  def _update_df(self, thresh_arr, chan_arr, frame, chan, im_dist, thresh, rad_arr):
    n_thresh = np.sum(thresh_arr)
    if np.any(thresh_arr):
      ones_vec = np.ones(n_thresh)
      x_vec = self.x_arr[thresh_arr].flatten()
      y_vec = self.y_arr[thresh_arr].flatten()
      fluor_vec = chan_arr[thresh_arr].flatten()
      frame_vec = frame*ones_vec
      chan_vec = chan*ones_vec
      pad_vec = self.pad_ind*ones_vec
      scale_vec = self.scale*ones_vec
      thresh_vec = thresh*ones_vec
      dist_vec = im_dist[thresh_arr].flatten()
      rad_vec = rad_arr[thresh_arr].flatten()
      data_cols = [frame_vec, x_vec, y_vec, pad_vec, fluor_vec, chan_vec,
                   scale_vec, thresh_vec, dist_vec, rad_vec]
      update_df = pd.DataFrame(dict(zip(self.columns, data_cols)))
      return update_df
    return None

  def process_frame(self, frame):
    ny, nx, nc = self.pad_helper.pad_arr.shape
    acq, pad_helper = self.acq, self.pad_helper
    # Prep frame data
    if self.print_img:
      printer = self.printer
      for ci in np.arange(0,nc):
        if self.diff:
          pad_helper.pad_arr[:,:,ci] = self.diff_arr[frame, ci,:,:]
        else:
          pad_helper.pad_arr[:,:,ci] = self.arr[frame, ci,:,:]
      pad_helper._prep_img_arr()
      pad_helper._prep_norm_arr()
      printer.ims[0].set_data(pad_helper.norm_arr)
      printer.axs[1].set_title(frame)

    out_df = pd.DataFrame(columns=self.columns, dtype=np.float)
    # Get distance array and prep frame data
    im_dist, y_thresh, rad_arr = self._yfp_dist(frame)
    if self.print_img:
      pad_helper.pad_arr[:,:,0] = im_dist==0

    # Run it
    for c_i in [0,1,2]:
      chan = acq.chan_ind_list[c_i]
      if self.diff:
        thresh_arr, chan_arr, thresh = self._chan_diff_thresh(frame, c_i)
      else:
        thresh_arr, chan_arr, thresh = self._chan_thresh(frame, c_i)
      if self.print_img:
        printer._prep_thresh_chan(im_dist, thresh_arr, c_i)
      update_df = self._update_df(thresh_arr, chan_arr, frame, chan, im_dist, thresh, rad_arr)
      if not (update_df is None):
        out_df = pd.concat([out_df, update_df], ignore_index=True)
    if self.print_img:
      printer._finish_thresh(frame)
      out_fn = self.png_tmpl.format(frame, self.pad_ind)
      printer.fig.savefig(out_fn)
    return out_df

  def begin(self):
    diff = self.diff
    # Print img setup
    out_df = pd.DataFrame(columns=self.columns, dtype=np.float)
    # Check that csvs directory exists
    out_dir = os.path.join(self.acq.super_dir,"csvs")
    if not os.path.isdir(out_dir):
      os.mkdir(out_dir)
    if self.print_img:
      self.printer = ComparisonAnimator(self)
      out_tmpl = 'frame{:02d}_pad{:02d}.png'
      if diff:
        out_tmpl = os.path.abspath(os.path.join('.','pngs','diff_thresh_progress', out_tmpl))
      else:
        out_tmpl = os.path.abspath(os.path.join('.','pngs','thresh_progress', out_tmpl))
      self.png_tmpl = out_tmpl
    if diff:
      out_tmpl = os.path.join(out_dir,"diff_pad{}.csv")
    else:
      out_tmpl = os.path.join(out_dir,"pad{}.csv")
    for frame in self.acq.frame_vec[:-1]:
      in_df = self.process_frame(frame)
      out_df = pd.concat([in_df, out_df], axis=0, ignore_index=True)
    n_rows = len(out_df)
    acqname = "_".join(self.acq.super_dir.split(os.path.sep)[-2:])
    acqname_vec = np.repeat(acqname, n_rows)
    acqname_col = pd.DataFrame(acqname_vec, index=out_df.index, columns=['acqname'])
    out_df = pd.concat([out_df, acqname_col], axis=1)
    out_df.to_csv(out_tmpl.format(self.pad_ind), index=False)

class FitHelper():
  def __init__(self, fit_t, fit_x, t0, gamma=100):
    self.fit_t = fit_t
    self.fit_x = fit_x
    self.t0 = t0
    self.gamma = gamma
    self.pe = self.p_est()

  def f_fun(self, p, t):
    t0,c0,c1,c2 = p
    t = t - t0
    y = np.zeros_like(t)
    y[t>0] = c0+np.power(c1*t[t>0],c2)
    y[t<=0] = 0
    return y

  def f_res(self, p):
    fit_t, fit_x = self.fit_t, self.fit_x
    t0,c0,c1,c2 = p
    if np.any(np.array([t0,c1,c2])<=0):
      return np.inf
    res = fit_x - self.f_fun(p, fit_t)
    return np.sum(np.log(np.power(res/self.gamma,2)+1))

  def est_res(self, p):
    fit_t, fit_x = self.fit_t, self.fit_x
    c0,c1 = p
    p = np.array([self.t0, c0, c1, 1])
    res = fit_x - self.f_fun(p, fit_t)
    return np.sum(np.power(res,2))

  def p_est(self):
    t, y = self.fit_t, self.fit_x
    c0,c1,c2 = 0,1/40,1
    #p0 = pe#np.array([t0,c1,c2])
    t = self.fit_t
    y = self.fit_x
    fit_out = opt.minimize(self.est_res, np.array([c0,c1]), method='Nelder-Mead',options={'maxiter':1000})
    c0, c1 = fit_out.x
#    y_diff = np.diff(y)
#    t_diff = np.diff(t)
#    dydt_vec = y_diff[t_diff!=0]/t_diff[t_diff!=0]
#    bool_vec = np.isfinite(dydt_vec)
#    c1 = np.mean(dydt_vec[bool_vec])
#    c0 = np.mean(y - self.f_fun((self.t0, c0,c1,c2), t))
    return np.array([self.t0, c0,c1,c2])

  def p_pad(self, p):
    return p
#    t0, c1, c2 = p
#    c0 = 0
#    t = self.fit_t
#    y = self.fit_x
#    sim_y = self.f_fun((t0,c0,c1,c2), t)
#    c0 = np.mean(y - sim_y)
#    p = np.array([t0,c0,c1,c2])
#    return p

  def res_wrapper(self, p):
    #return self.f_res(self.p_pad(p))
    return self.f_res(p)

  def fit_wrapper(self):
    pe = self.pe
    t0,c0,c1,c2 = pe
    p0 = pe#np.array([t0,c1,c2])
    t = self.fit_t
    y = self.fit_x
    fit_out = opt.minimize(self.res_wrapper, p0, method='Nelder-Mead',options={'maxiter':1000})
    if fit_out.success:
      p_out = self.p_pad(fit_out.x)
    else:
      #p_out = pe.copy()
      p_out = self.p_pad(fit_out.x)
      #print(fit_out)
    return fit_out.success, p_out

class FrontFitter():
  def __init__(self, acq, bin_width=None, chan_bin_widths=[250,500,2500], diff=False):
    out_dir = os.path.join(acq.super_dir,"csvs")
    if diff:
      self.csv_tmpl = os.path.join(out_dir,"diff_pad{}.csv")
      chan_bin_factor = 10
    else:
      self.csv_tmpl = os.path.join(out_dir,"pad{}.csv")
      chan_bin_factor = 1

    if bin_width is None:
      bin_width = 2*acq.pixel_size
    self.bin_width = bin_width
    #self.chan_bin_widths = chan_bin_widths
    chan_labs = acq.chan_ind_list
    chan_bin_widths = np.array(chan_bin_widths) / chan_bin_factor
    self.chan_bin_dict = dict(zip(chan_labs, chan_bin_widths))
    #self.quants = [low_quant, high_quant]
    self.acq = acq
    self.time_df = acq.time_df
    self.diff = diff

  def pad_peaks(self, pad_ind):
    bin_width = self.bin_width
    pad_df = pd.read_csv(self.csv_tmpl.format(pad_ind))
    d_vec = pad_df.dist.values
    pad_df['dist_binned'] = (d_vec//bin_width)*bin_width + (bin_width/2)
    inds = [(frame, pad_ind) for frame in pad_df.frame.values]
    t_vec = self.time_df.loc[inds,'time'].values
    pad_df['time'] = t_vec
    gb_chan = pad_df.groupby('channel')
    out_list = []
    for chan, chan_df in gb_chan:
      peak_df = self.chan_peaks(chan_df)
      if peak_df is None:
        continue
      peak_df['pad'] = pad_ind
      out_list.append(peak_df)
    if len(out_list) > 1:
      return pd.concat(out_list, ignore_index=False)
    else:
      return None

  def chan_peaks(self, chan_df):
    chan = chan_df.channel.values[0]
    pad_ind = chan_df['pad'].values[0]
    gb_cols = ['fluor','frame','dist_binned']
    fluor_gb_frame_db = chan_df[gb_cols].groupby(gb_cols[1:])
    q_lamb = lambda x : np.quantile(x, 0.95)
    agg_df = fluor_gb_frame_db.agg(q_lamb).reset_index()
    piv_df = agg_df.pivot(index='dist_binned', values='fluor', columns='frame')
    arr = piv_df.values
    arr_h, arr_w = arr.shape
    #vmin, vmax = [np.quantile(agg_df.fluor.values, xx) for xx in self.quants]
    chan_bin = self.chan_bin_dict[chan]
    peak_df = self.peak_finder(agg_df, chan_bin)#, vmin, vmax)
    if peak_df is None:
      return None
    peak_df['channel'] = chan
    peak_df['acqname'] = chan_df.acqname.values[0]
    inds = [(frame, pad_ind) for frame in peak_df.frame.values]
    t_vec = self.time_df.loc[inds,'time'].values
    peak_df['time'] = t_vec
    return peak_df

  def peak_finder(self, agg_df, chan_bin):#, vmin, vmax):
    out_list = []
    gb_cols = ['frame','dist_binned']
    #itr = zip(('low','high'),(vmin, vmax))
    n_bins = (agg_df.fluor.max()-agg_df.fluor.min())//chan_bin
    thresh_vec = agg_df.fluor.min() + np.arange(n_bins)*chan_bin
    for label, thresh in enumerate(thresh_vec):
      bool_vec = agg_df.fluor >= thresh
      sub_df = agg_df.loc[bool_vec, gb_cols].groupby(gb_cols[-1]).agg(np.min).reset_index()
      d_vec, frame_vec = sub_df.dist_binned.values, sub_df.frame.values
      peaks = signal.find_peaks(-1*frame_vec)[0]
      sub_df['level'] = label
      sub_df['thresh'] = thresh
      out_list.append(sub_df.loc[peaks,:])
    if len(out_list) > 1:
      return pd.concat(out_list, ignore_index=True)
    else:
      return None

  def fit_wrapper(self, peak_df):
    columns = ['t0','c0','c1','c2','success','pad','channel','level']
    pad_ind = peak_df['pad'].values[0]
    inds = [(frame, pad_ind) for frame in peak_df.frame.values]
    t_vec = self.time_df.loc[inds,'time'].values
    peak_df['time'] = t_vec
    gb_level = peak_df.groupby(['channel','level'])
    out_list = []
    for key, sub_df in gb_level:
      chan, level = key
      sub_df = sub_df.sort_values(by='time')
      fithelper = FitHelper(sub_df.time, sub_df.dist_binned, 1e4)
      self.fithelper = fithelper
      fit_success, p_fit = fithelper.fit_wrapper()
      if not fit_success:
        p_fit = fithelper.p_est()
      t0,c0,c1,c2 = p_fit
      d_vec = np.array([[t0,c0,c1,c2,fit_success,pad_ind,chan,level]]).T
      df_row = pd.DataFrame(dict(zip(columns, d_vec)))
      out_list.append(df_row)
    out_df = pd.concat(out_list, ignore_index=True)
    float_cols = ['t0','c0','c1','c2','channel','pad']
    for col in float_cols:
      out_df[col] = out_df[col].values.astype(np.float)
    int_cols = ['channel','pad']
    for col in int_cols:
      out_df[col] = out_df[col].values.astype(np.int)
    return out_df

  def proc_all_pads(self):
    out_list = []
    peak_out_list = []
    for pad in np.unique(self.acq.cor_pos_df['pad'].values):
      peak_df = self.pad_peaks(pad)
      if peak_df is None:
        continue
      peak_out_list.append(peak_df)
      if len(peak_df) > 0:
        out_list.append(self.fit_wrapper(peak_df))
    return pd.concat(out_list, ignore_index=True), pd.concat(peak_out_list, ignore_index=True)

with open('txt_files/unique_good_dirs.txt', 'r') as f:
    lines = f.read().splitlines()
