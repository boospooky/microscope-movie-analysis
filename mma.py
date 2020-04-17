# James Parkin. Murray biocircuits lab. 2020.
# Attempt to create a suite of python functions and classes to aid in the 
# visualization and analysis of microscope movies on the HPC.

# TODO
# Convert notebook code for file accounting into functions

# OVERVIEW
# Module variables
# 
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
import scipy as scp
import pandas as pd

import matplotlib.animation as anm
from IPython.display import HTML

from skimage.external.tifffile import TiffWriter

from PIL import Image, UnidentifiedImageError

import skimage.filters
import skimage.io
import skimage.morphology
import skimage.feature
import skimage.exposure

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

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
'190727_pfbs/img_2':(np.arange(70,82)),
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
'190725_osc2/data':True,
'190726_i52_2/ladder_img_1':True,
'190727_pfbs/img_2':True,
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
'20190122_small_rhl_pads/20190122_1':True,
'20190122_small_rhl_pads/20190122_b_1':True}

pad_rotation_dict = {'180312_full_circuit_w_senders/20180302_2':0,
'180313_full_circuit_w_senders/20180313_1':0,
'180314_full_circuit_w_senders/20180314_1':0,
'180316/20180316_fullcircuit_longpads_1':0,
'180320/180320_fullcircuit_longpads_1':0,
'180601/180601_1':0,
'180711/20180711_1':0,
'180822_movie/20180822_1':0,
'181222_movie/20181222_1':0,
'190126_rhl_strain_6/20190126_2':0,
'190126_rhl_strain_6/20190126_4':0,
'190221_inducer_test/20190221_1':0,
'190222_pfb_test/20190222_1':0,
'190725_osc2/data':0,
'190726_i52_2/ladder_img_1':0,
'190727_pfbs/img_2':0,
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

    # Each frame should be associated with the same number of images. 
    # If this is not the case, the acquisition was aborted.
    # Remove rows corresponding to the frames with too few images
#    gb_frame = files_df.groupby('frame')
#    imgs_per_frame = files_df[['frame','fn']].groupby('frame').agg(len)
#    max_ipf = imgs_per_frame['fn'].max()
#    short_frames = imgs_per_frame.index[imgs_per_frame.fn < max_ipf]
#    for frame_i in short_frames:
#        drop_inds = gb_frame.get_group(frame_i).index
#        files_df.drop(drop_inds, axis=0, inplace=True)
#        gb_frame = files_df.groupby('frame')
#    files_df = files_df.reindex()
#    n_files = len(files_df)
#    # pad indices inferred from file paths may not be contiguous
#    # replace with order vector to ensure 0-indexed pads
#    gb_pad = files_df.groupby('pad')
#    for i, inds in enumerate(gb_pad.groups.values()):
#        files_df.loc[inds,'pad'] = i

    # Assign integer values to each position. If any are missed, throw exception
    gb_pos = files_df.groupby(['posname'])
    for i, inds in enumerate(gb_pos.groups.values()):
        files_df.loc[inds,'pos'] = i
    assert not np.any(files_df.pos.values == -1)
    res = apply_manual_pad_labeling(files_df, dict_key)
    if np.any(files_df.pad.values<0):
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
      pad_ind = files_df.pad.max()+1
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
    n_pads = len(np.unique(files_df.pad))
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

def plot_positions(cor_pos_df, out_fn):
    plt.figure(figsize=(18,18))
    n_pads = len(cor_pos_df.groupby('pad'))
    colors = sns.color_palette('Set1', n_colors=n_pads)
    for p_i in cor_pos_df.index:
        point_color = colors[np.int(cor_pos_df.loc[p_i,"pad"])]
        plt.plot(np.float(cor_pos_df.x[p_i]),
                 -np.float(cor_pos_df.y[p_i]),
                 '.',
                 label=cor_pos_df.pad[p_i],
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
    plt.savefig(out_fn)
    plt.close('all')

def write_movie_no_bg(out_fn, pad_ind, files_df, cor_pos_df, pixel_size, scale=4, rotation=180):
    # Get movie metadata
    n_chan, chan_ind_list, chan_names, im_width, im_height = get_exp_summary_from_fn(files_df.fn.values[0])
    n_frames = files_df.frame.max()+1
    frame_vec = np.sort(np.unique(files_df.frame.values))
    pixel_size = pixel_size * scale
    h, w = im_height//scale, im_width//scale
    uint_max = 65535
    # Setup plotting variables
    plt.close('all')
    pad_df = cor_pos_df.loc[cor_pos_df.pad==pad_ind,:]
    pos_list = np.unique(pad_df.index.values)
    xlims = np.array([pad_df.x.min(), pad_df.x.max()])/pixel_size + w*np.array([-0.25,1.25])
    ylims = np.array([pad_df.y.min(), pad_df.y.max()])/pixel_size + h*np.array([-0.25,1.25])
    rel_mins = np.concatenate([[xlims], [ylims]],axis=0).astype(np.int)
    pad_h = np.int(np.ceil(np.diff(ylims)))
    pad_w = np.int(np.ceil(np.diff(xlims)))
    fig_h, fig_w = np.array([pad_h, pad_w])/(100/scale)
    while fig_h > 4e3 or fig_w > 4e3:
        fig_h, fig_w = np.array([fig_h, fig_w])/10
    # Define helper functions
    def load_img(frame, pos, channel, rotation=rotation, scale=scale):
        frame_bool = files_df.frame==frame
        pos_bool = files_df.pos==pos
        chan_bool = files_df.channel==channel
        indx_bool = (frame_bool)&(pos_bool)&(chan_bool)
        if sum(indx_bool) < 1:
            return skimage.transform.rotate(np.zeros((h,w),dtype=np.float32),rotation)
            error_msg = 'Image not found when printing {}: pad:{} frame:{} pos:{} channel:{}'
            error_msg = error_msg.format(out_fn, pad_ind, frame, pos, channel)
            print(error_msg)
        fn = files_df.loc[indx_bool,'fn'].values[0]
        try:
            img = skimage.io.imread(fn).astype(np.float32)
        except ValueError as error:
            msg = 'skimage img read error {}'.format(fn)
            print(msg)
            debug_log(error, msg)
            return skimage.transform.rotate(np.zeros((h,w), dtype=np.float32),rotation)
        # img = skimage.filters.gaussian(img, sigma)
        img = skimage.transform.downscale_local_mean(img, (scale, scale))
        return skimage.transform.rotate(img, rotation)

    pad_arr = np.zeros((pad_h, pad_w, n_chan), dtype=np.float32)
    chan_vec = np.array(chan_ind_list)
    pos_lims = (pad_df[['x','y']]/pixel_size).astype(np.int)
    count_arr = np.zeros((pad_h, pad_w), dtype=np.float32)
    img_arr = skimage.transform.rotate(np.ones((h,w),np.float32),rotation)
    for pos in pos_list:
        x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
        count_arr[y0:y0+h,x0:x0+w] += img_arr
    uncovered_arr = count_arr<=0
    covered_arr = count_arr>0
    count_arr[count_arr<=0] = 1

    def get_frame_arr(frame_ind):
        im_arr = np.zeros((h,w),dtype=np.float32)
        for chan_i, channel in enumerate(chan_vec):
            for pos in pos_list:
                x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
                count_portion = count_arr[y0:y0+h,x0:x0+w]
                im_arr[:] = 0
                im_arr += load_img(frame_ind, pos, channel)
                pad_arr[y0:y0+h,x0:x0+w,chan_i] += im_arr # - bg_arr
            pad_arr[:,:,chan_i] /= count_arr

    def animate(i):
        img_arr = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        pad_arr[:] = 0
        get_frame_arr(i)
        # For no-bg gifs, normalize relative to the middle frame.
        # Use contrast-stretching approach, where the bounds are the middle 90 percentile
        norm_vec = []
        for chan_ind in np.arange(n_chan):
            frame_vals = pad_arr[:,:,chan_ind][covered_arr].flatten()
            frame_vals = np.sort(frame_vals)
            frame_n = len(frame_vals)
            ind_min, ind_max = np.array([0.05*frame_n, 0.95*frame_n],dtype=np.int)
            vmin, vmax = frame_vals[ind_min], frame_vals[ind_max]
            vmax = np.max([vmax+vmin,vmin*2])
            norm_fn = mpl_colors.Normalize(vmin, vmax, clip=True)
            norm_vec.append(norm_fn)
        for chan_ind, chan_slot in enumerate(chan_vec):
            norm = norm_vec[chan_ind]
            chan_arr = pad_arr[:,:,chan_ind:chan_ind+1]
            color_vec = mpl_colors.to_rgb(mpl_named_colors[chan_slot])
            img_arr += np.concatenate([norm(chan_arr)*color_val for color_val in color_vec],axis=2)
        norm_fn = mpl_colors.Normalize(0, 1, clip=True)
        im.set_array(norm_fn(img_arr))

    fig, ax = plt.subplots(1, 1, figsize=(fig_w,fig_h+1.2))
    img_arr = np.zeros((pad_h, pad_w, 3),dtype=np.float32)
    im = ax.imshow(img_arr.copy(), animated=True, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pad {}".format(pad_ind))

    fig.tight_layout()

    # animate(frame_ind)
    anim = anm.FuncAnimation(fig, animate, interval=100, frames=frame_vec)
    plt.close('all')
    anim.save(out_fn, dpi=80, fps=3, writer='pillow')

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
    for pos in pos_list:
        x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
        count_arr[y0:y0+h,x0:x0+w] += self.one_arr
    pos_lims = (pad_df[['x','y']]/pixel_size).astype(np.int)
    xlims = np.array([pad_df.x.min(), pad_df.x.max()])/pixel_size + w*np.array([-0.25,1.25])
    ylims = np.array([pad_df.y.min(), pad_df.y.max()])/pixel_size + h*np.array([-0.25,1.25])
    rel_mins = np.concatenate([[xlims], [ylims]],axis=0).astype(np.int)
    pad_h = np.int(np.ceil(np.diff(ylims)))
    pad_w = np.int(np.ceil(np.diff(xlims)))
    fig_h, fig_w = np.array([pad_h, pad_w])/(1000/scale)
    while fig_h > 1e4 or fig_w > 1e4:
        fig_h, fig_w = np.array([fig_h, fig_w])/10
    count_arr = np.zeros((pad_h, pad_w),dtype=np.float32)
    self.count_arr,self.uncovered_arr,self.covered_arr = count_arr,count_arr<=0,count_arr>0
    self.count_arr[self.count_arr<=0] = 1
    self.h, self.w = h, w
    self.fig_h, self.fig_w = fig_h, fig_w
    self.pixel_size = pixel_size
    self.rel_mins,self.pos_list,self.pos_lims = rel_mins,pos_list,pos_lims
    self.pad_arr, self.img_arr = np.zeros((pad_h, pad_w, n_chan),dtype=np.float32), np.zeros((pad_h, pad_w, 3),dtype=np.float32)

  def __init__(self, acq, scale=8, pad_ind=None):
    if pad_ind is None:
      pad_ind = 0
    # Get movie metadata
    self.acq = acq
    uint_max = 65535
    # Setup plotting variables
    self.pad_df = cor_pos_df.groupby('pad').get_group(pad_ind)
    self.chan_vec = acq.chan_ind_list
    self.pixel_size=acq.pixel_size
    self.scale,self.rotation = scale,acq.rotation

  def load_img(self, frame, pos, channel):
    frame_bool = self.files_df.frame==frame
    pos_bool = self.files_df.pos==pos
    chan_bool = self.files_df.channel == channel
    indx_bool = (frame_bool)&(pos_bool)&(chan_bool)
    if sum(indx_bool) < 1:
      return skimage.transform.rotate(np.zeros((self.h,self.w),dtype=np.float32), self.rotation)
      error_msg = 'Image not found when printing {}: pad:{} frame:{} pos:{} channel:{}'
      error_msg = error_msg.format(self.out_fn, frame, pos, channel)
      print(error_msg)
    fn = self.files_df.loc[indx_bool,'fn'].values[0]
    try:
      img = skimage.io.imread(fn).astype(np.float32)
    except Exception as error:
      print('import img error {}'.format(fn))
      raise error
    img = skimage.transform.downscale_local_mean(img, (self.scale, self.scale))
    return skimage.transform.rotate(img, self.rotation, resize=True)

  def get_frame_arr(self, frame_ind):
    h, w = self.one_arr.shape
    self.pad_arr[:] = 0
    frame_arr = np.zeros((h, w), dtype=np.float32)
    for chan_i, channel in enumerate(self.chan_vec):
      for pos in self.pos_list:
        x0, y0 = (self.pos_lims.loc[pos,:].values - self.rel_mins[:,0]).astype(np.int)
        yslice = slice(y0,y0+h)
        xslice = slice(x0,x0+w)
        frame_arr[:] = 0
        frame_arr += self.load_img(frame_ind, pos, channel)
        self.pad_arr[yslice,xslice,chan_i] += frame_arr / self.count_arr[yslice, xslice]

  def animate(self, frame_ind):
    pad_h, pad_w = self.pad_arr.shape[:2]
    self.img_arr[:] = 0
    self.get_frame_arr(frame_ind)
    # Use contrast-stretching approach, where the bounds are the middle 90 percentile
    norm_vec = []
    for chan_ind in np.arange(self.n_chan):
      frame_vals = self.pad_arr[:,:,chan_ind][self.covered_arr].flatten()
      frame_vals = np.sort(frame_vals)
      n_vals = len(frame_vals)
      ind_min, ind_max = np.array([0.05*n_vals, 0.95*n_vals],dtype=np.int)
      vmin, vmax = frame_vals[ind_min], frame_vals[ind_max]
      vmax = np.max([vmax+vmin,vmin*2])
      norm_fn = mpl_colors.Normalize(vmin, vmax, clip=True)
      norm_vec.append(norm_fn)
    for chan_ind, chan_slot in enumerate(self.chan_vec):
      norm = norm_vec[chan_ind]
      chan_arr = self.pad_arr[:,:,chan_ind:chan_ind+1]
      color_vec = mpl_colors.to_rgb(mpl_named_colors[chan_slot])
      self.img_arr += np.concatenate([norm(chan_arr)*color_val for color_val in color_vec],axis=2)
    norm_fn = mpl_colors.Normalize(0, 1, clip=True)
    normed_arr = (255*norm_fn(self.img_arr)).astype(np.uint8)
    self.im.set_array(normed_arr)

  def setup_plot(self):
    fig, ax = plt.subplots(1, 1, figsize=(self.fig_w, self.fig_h+1.2))
    self.im = ax.imshow(np.zeros((self.h, self.w), dtype=np.uint8), animated=True, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(self.out_fn)
    fig.tight_layout()
    return fig, ax

  def save_anim(self, out_fn, writer='ffmpeg'):
    self.out_fn = out_fn
    fig, ax = self.setup_plot()
    anim = anm.FuncAnimation(fig, self.animate, interval=100, frames=self.frame_vec)
    plt.close('all')
    anim.save(out_fn, dpi=80, fps=3, writer=writer)

  def save_frame(self, out_fn, frame_ind, writer):
    self.out_fn = out_fn
    fig, ax = self.setup_plot()
    self.animate(frame_ind)
    fig.savefig(out_fn)
    plt.close('all')

def write_frame_no_bg(frame_ind, out_fn, pad_ind, files_df, cor_pos_df, pixel_size, scale=4, rotation=180):
    # Get movie metadata
    n_chan, chan_ind_list, chan_names, im_width, im_height = get_exp_summary_from_fn(files_df.fn.values[0])
    n_frames = files_df.frame.max()+1
    pixel_size = pixel_size * scale
    h, w = im_height//scale, im_width//scale
    uint_max = 65535
    # Setup plotting variables
    plt.close('all')
    pad_df = cor_pos_df.loc[cor_pos_df.pad==pad_ind,:]
    pos_list = np.unique(pad_df.index.values)
    xlims = np.array([pad_df.x.min(), pad_df.x.max()])/pixel_size + w*np.array([-0.25,1.25])
    ylims = np.array([pad_df.y.min(), pad_df.y.max()])/pixel_size + h*np.array([-0.25,1.25])
    rel_mins = np.concatenate([[xlims], [ylims]],axis=0).astype(np.int)
    pad_h = np.int(np.ceil(np.diff(ylims)))
    pad_w = np.int(np.ceil(np.diff(xlims)))
    fig_h, fig_w = np.array([pad_h, pad_w])/(1000/scale)
    while fig_h > 1e4 or fig_w > 1e4:
        fig_h, fig_w = np.array([fig_h, fig_w])/10
    # Define helper functions
    def load_img(frame, pos, channel, rotation=rotation, scale=scale, sigma=3):
        frame_bool = files_df.frame==frame
        pos_bool = files_df.pos==pos
        chan_bool = files_df.channel == channel
        indx_bool = (frame_bool)&(pos_bool)&(chan_bool)
        if sum(indx_bool) < 1:
            return skimage.transform.rotate(np.zeros((h,w)),rotation)
            error_msg = 'Image not found when printing {}: pad:{} frame:{} pos:{} channel:{}'
            error_msg = error_msg.format(out_fn, pad_ind, frame, pos, channel)
            print(error_msg)
        fn = files_df.loc[indx_bool,'fn'].values[0]
        try:
            img = skimage.filters.gaussian(skimage.io.imread(fn).astype(np.float), sigma)
        except Exception as error:
            print('import img error {}'.format(fn))
            raise error
        img = skimage.transform.downscale_local_mean(img, (scale, scale))
        return skimage.transform.rotate(img, rotation)

    pad_arr = np.zeros((pad_h, pad_w, n_chan))
    chan_vec = np.array(chan_ind_list)
    pos_lims = (pad_df[['x','y']]/pixel_size).astype(np.int)
    count_arr = np.zeros((pad_h, pad_w))
    img_arr = skimage.transform.rotate(np.ones((h,w)),rotation)
    for pos in pos_list:
        x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
        count_arr[y0:y0+h,x0:x0+w] += img_arr
    uncovered_arr = count_arr<=0
    covered_arr = count_arr>0
    count_arr[count_arr<=0] = 1

    def get_frame_arr(frame_ind):
        # bg_arr = np.zeros((h,w))
        im_arr = np.zeros((h,w))
        # get background image
        # bg_pos_list = [83,89,95,101,107]
        # n_bg = len(bg_pos_list)
        for chan_i, channel in enumerate(chan_vec):
            # bg_arr[:] = 0
            # bg_arr += np.mean([load_img(frame_ind, pos, channel) for pos in bg_pos_list], axis=0)
            for pos in pos_list:
                x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
                im_arr[:] = 0
                im_arr += load_img(frame_ind, pos, channel)
                pad_arr[y0:y0+h,x0:x0+w,chan_i] += im_arr # - bg_arr
            pad_arr[:,:,chan_i] /= count_arr

    def animate(i):
        img_arr = np.zeros((pad_h, pad_w, 3))
        pad_arr[:] = 0
        get_frame_arr(i)
        # For no-bg gifs, normalize relative to the middle frame.
        # Use contrast-stretching approach, where the bounds are the middle 90 percentile
        norm_vec = []
        for chan_ind in np.arange(n_chan):
            frame_vals = pad_arr[:,:,chan_ind][covered_arr].flatten()
            frame_vals = np.sort(frame_vals)
            frame_n = len(frame_vals)
            ind_min, ind_max = np.array([0.05*frame_n, 0.95*frame_n],dtype=np.int)
            vmin, vmax = frame_vals[ind_min], frame_vals[ind_max]
            vmax = np.max([vmax+vmin,vmin*2])
            norm_fn = mpl_colors.Normalize(vmin, vmax, clip=True)
            norm_vec.append(norm_fn)
        for chan_ind, chan_slot in enumerate(chan_vec):
            norm = norm_vec[chan_ind]
            chan_arr = pad_arr[:,:,chan_ind:chan_ind+1]
            color_vec = mpl_colors.to_rgb(mpl_named_colors[chan_slot])
            img_arr += np.concatenate([norm(chan_arr)*color_val for color_val in color_vec],axis=2)
        im.set_array(img_arr)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w,fig_h+1.2))
    img_arr = np.zeros((pad_h, pad_w, 3))
    im = ax.imshow(img_arr.copy(), animated=True, interpolation='none')
    #     cbar = fig.colorbar(im, ax=ax, ticks=[vmin, vmax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pad {}".format(pad_ind))

    fig.tight_layout()

    animate(frame_ind)
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
            fn = self.files_df.fn.values[0]
            all_metadata = fn_metadata_full(fn)
            summ_dict = all_metadata['Summary']
            self.chan_ind_list = [int(xx) for xx in summ_dict['ChNames']]
            self.chan_names = [all_channel_names[xx] for xx in self.chan_ind_list]
            self.n_chan = len(self.chan_ind_list)
            self.im_width = summ_dict['Width']
            self.im_height = summ_dict['Height']
            self.binning = all_metadata['Binning']
            self.objective = all_metadata['Objectives-State']
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
        plot_positions(self.cor_pos_df, out_fn)

    def _prep_writehelper(self, ):
        args = pad_ind, self.files_df, self.cor_pos_df, self.pixel_size, scale, self.rotation

    def write_movie_class(self, pad_ind, scale=8, skip=1):
        out_fn = '{}_{}_pad_{}.gif'.format(self.expname, self.acqname, pad_ind)
        # out_fn = os.path.join(self.super_dir, out_fn)
        out_fn = os.path.abspath(os.path.join('.','anims', out_fn))
        args = out_fn, pad_ind, self.files_df, self.cor_pos_df, self.pixel_size, scale, self.rotation
        try:
            self.writer_obj = WriteHelper(*args)
            self.writer_obj.save_anim(out_fn, 'pillow')
        except FileNotFoundError as err:
            with open('bad_files.txt','a') as f:
                f.write(self.super_dir+"\n")
                print(err)

    def write_all_pad_gifs_class(self, scale=8, skip=1):
        pad_inds = np.unique(self.files_df.pad.values)
        for pad_ind in pad_inds:
            self.write_movie_class(pad_ind, scale, skip)

    def write_movie_no_bg(self, pad_ind, scale=8, skip=1):
        out_fn = '{}_{}_pad_{}.gif'.format(self.expname, self.acqname, pad_ind)
        # out_fn = os.path.join(self.super_dir, out_fn)
        out_fn = os.path.abspath(os.path.join('.','anims', out_fn))
        args = out_fn, pad_ind, self.files_df, self.cor_pos_df, self.pixel_size, scale, self.rotation
        try:
            write_movie_no_bg(*args)
        except FileNotFoundError as err:
            with open('bad_files.txt','a') as f:
                f.write(self.super_dir+"\n")
                print(err)

    def write_all_pad_gifs_no_bg(self, scale=8, skip=1):
        pad_inds = np.unique(self.files_df.pad.values)
        for pad_ind in pad_inds:
            self.write_movie_no_bg(pad_ind, scale, skip)

    def write_frame_no_bg(self, frame_ind, pad_ind, scale=8):
        out_fn = '{}_{}_pad_{}_frame_{}.avi'.format(self.expname, self.acqname, pad_ind, frame_ind)
        #out_fn = os.path.join(self.super_dir, out_fn)
        out_fn = os.path.abspath(os.path.join('.','pngs', out_fn))
        print(out_fn)
        args = frame_ind, out_fn, pad_ind, self.files_df, self.cor_pos_df, self.pixel_size, scale, self.rotation
        try:
            write_frame_no_bg(*args)
        except FileNotFoundError as err:
            with open('bad_files.txt','a') as f:
                f.write(self.super_dir+"\n")
                print(err)

    def write_all_pad_frame_no_bg(self, frame_ind, scale=8):
        pad_inds = np.unique(self.files_df.pad.values)
        for pad_ind in pad_inds:
            self.write_frame_no_bg(frame_ind, pad_ind, scale)

def get_bg_img(frame, channel, files_df):
    gb_pad = files_df.groupby(('pad','frame','channel'))
    gb_pos = files_df.groupby(('pos','frame','channel'))
    fns = gb_pad.get_group((6,frame,channel)).fn
    imgs = [skimage.io.imread(fn) for fn in fns]
    h, w = imgs[0].shape
    bg_im = np.median(np.concatenate([img.reshape((h, w, 1)) for img in imgs],axis=2),axis=2)
    return bg_im

def label_helper(im_arr, bg_im_arr):
    uint_max = 65535
    im_arr = im_arr / uint_max
    bg_im_arr = bg_im_arr / uint_max
    w, h = im_arr.shape
    # Smooth to reduce noise
    g_radius = 5
    im_smooth = skimage.filters.gaussian(im_arr, g_radius)
    bg_smooth = skimage.filters.gaussian(bg_im_arr, g_radius)
    im_bgsub = im_smooth - bg_smooth
    im_bgsub[im_bgsub < 0] = 0

    thresh = skimage.filters.threshold_li(im_bgsub)
    thresh = np.max([40/uint_max,thresh])
    im_bw = im_bgsub > thresh
    if np.sum(im_bw) == 0 :
        return 0, 0
    im_labeled, num = skimage.morphology.label(im_bw, return_num=True)
    return im_labeled, num

def process_img(df_row, cor_pos_df):
    bg_tmpl = 'worker_outputs/bg_arr_{}_{}.npy'
    pad, pos, fn, frame, channel, im_time = df_row[['pad','pos','fn','frame','channel','time']].values
    py, px = cor_pos_df.loc[pos,['y', 'x']]
    im_arr = skimage.io.imread(fn)
    bg_im_arr = np.load(bg_tmpl.format(frame, channel))
    im_bgsub = im_arr - bg_im_arr
    im_labeled, num = label_helper(im_arr, bg_im_arr)
    columns = ['area', 'x','y','fluor','label','time']
    if num == 0 :
        return pd.DataFrame(columns=columns)
    # make df
    index = np.arange(num)
    df = pd.DataFrame(columns=columns,index=index)
    regionprops_list = skimage.measure.regionprops(im_labeled)
    for label in np.arange(1,num+1):
        regionprops = regionprops_list[label-1]
        mask = im_labeled == label
        fluor = np.mean(mask*im_bgsub)
        y, x = np.array((py,px)) - np.array(regionprops.centroid)*um_pix
        df.loc[label-1,columns] = regionprops.area, x, y, fluor, label, im_time
    return df

def process_df(fn_out, sub_df, cor_pos_df):
    dfs = [process_img(sub_df.iloc[xx,:], cor_pos_df) for xx in np.arange(len(sub_df))]
    out_df = pd.concat(dfs, ignore_index=False)
    out_df.to_csv(fn_out)

def par_worker(args):
    process_df(*args)

def process_all(files_df, cor_pos_df):
    n_proc = os.cpu_count()
    n_rows = len(files_df)
    sub_df_list = [files_df.iloc[i::n_proc,:].copy() for i in np.arange(n_proc)]
    out_files = ['par_sub_{}.csv'.format(i) for i in np.arange(n_proc)]
    with multiprocessing.Pool(n_proc) as pool:
#         pool.map(par_worker, zip(out_files, sub_df_list, [cor_pos_df.copy() for xx in np.arange(n_proc)]))
        jobs = []
        for out_fn, sub_df in zip(out_files, sub_df_list):
            res = pool.apply_async(process_df, args=(out_fn, sub_df.copy(), cor_pos_df.copy()))
            jobs.append(res)
        pool.close()
        pool.join()
    return out_files

with open('unique_good_dirs.txt', 'r') as f:
    lines = f.read().splitlines()

