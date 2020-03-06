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

import os
import sys
import time
import datetime
import re
import multiprocessing
import ast

import numpy as np
import matplotlib.colors as mpl_colors
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

from PIL import Image

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

# Functions
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
    metadata_str = img.tag[metadata_key].decode('utf-8').replace("\x00","")
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
    img_files = [os.path.join(pos_dir, xx) for pos_dir in pos_dirs for xx in os.listdir(pos_dir) if '.tif' in xx]
    return img_files

def make_timevec(img_files):
    ctime_parallel_wrapper(img_files)
    time_vec = ctime_parallel_read_result(img_files)
    time_vec = time_vec - time_vec.min()
    np.save('timevec.npy',time_vec)

def create_filesdf(super_dir):
    img_files = find_tiffs_in_dir(super_dir)
    make_timevec(img_files)
    time_vec = np.load('timevec.npy')

    # Extract movie metadata
    all_metadata = fn_metadata_full(img_files[0])
    summ_dict = all_metadata['Summary']
    chan_ind = summ_dict['ChNames']
    chan_int = [int(xx) for xx in chan_ind]
    n_chan = len(chan_ind)
    chan_int_dict = dict(zip(chan_int, np.arange(n_chan)))
    chan_name = [all_channel_dict[xx] for xx in chan_ind]
    n_pos = len(pos_dirs)#summ_dict['Positions']
    im_width = summ_dict['Width']
    im_height = summ_dict['Height']

    # Interpret pad, frame, channel from filenames
    img_re_pattern = r'{}/(?P<pad>[0-9]+)-Pos_(?P<padcol>[0-9]+)_(?P<padrow>[0-9]+)/img_(?P<frame>[0-9]*)_(?P<channel>[0-9]*)_000.tif'.format(super_dir)
    rem = re.findall(img_re_pattern, '\n'.join(img_files), re.MULTILINE)
    rem_arr = np.array(rem)
    rem_arr[rem_arr==''] = -1
    metadata_arr = rem_arr.astype(np.int)
    n_rows = metadata_arr.shape[0]
    pad_vec, padcol_vec, padrow_vec, frame_vec, channel_vec = metadata_arr.T
    # padpos_vec = padpos_vec1 + padpos_vec2
    pad_vec = pad_vec - np.min(pad_vec)
    time_vec = np.load('timevec.npy')
    pos_vec = np.empty(n_rows,dtype=np.int)
    padpos_vec = np.empty(n_rows,dtype=np.int)
    n_frames = np.max(frame_vec)+1

    # Make dataframe
    columns = ['pos','pad','padcol','padrow','frame','channel','fn','time']
    col_data = [pos_vec, pad_vec, padcol_vec, padrow_vec, frame_vec, channel_vec, img_files, time_vec]
    n_rows = len(img_files)
    n_pads = len(np.unique(pad_vec))
    n_pos = np.int(n_rows/n_frames/n_chan)
    files_df = pd.DataFrame(dict(zip(columns, col_data)))

    files_df.sort_values(by='time', inplace=True)
    # TODO : reintroduce check that each position has the same number of frames

    gb_pp = files_df.groupby(['padcol','padrow'])
    for i, inds in enumerate(gb_pp.groups.values()):
        files_df.loc[inds,'padpos'] = i
    files_df.loc[:,'pos'] = np.tile(np.repeat(np.arange(n_pos), n_chan), n_frames)

    pos_df = files_df.loc[:,['pos','pad']].groupby('pos').agg(np.min)
    files_df.to_csv('filesdf.csv',index=False)

def make_positions_df():
    files_df = pd.read_csv('filesdf.csv')
    gb_frame = files_df.groupby('frame')
    sub_len = np.array([xx[1].shape[0] for xx in gb_frame])
    frames = np.array([xx for xx in gb_frame.groups])

    max_frame = frames[1:][np.diff(sub_len)==0].max()
    files_df = files_df.loc[files_df.frame<=max_frame,:]

    n_pads = len(np.unique(files_df.pad))
    n_frames = np.max(files_df.frame)+1
    pos_df = files_df.loc[:,['pos','pad']].groupby('pos').agg(np.min)

    label_vec = [xx['Label'] for xx in summ_dict['InitialPositionList']]
    xy_vec = [xx['DeviceCoordinatesUm']['XYStage'] for xx in summ_dict['InitialPositionList']]
    label_xy_dict = dict(zip(label_vec, xy_vec))
    pos_file_df = files_df.groupby('pos').agg(np.min)

    # Cor_pos_df is corrected position DF. In this notebook, correcting the position is not
    # necessary as there are only two imaging positions per pad. DF also includes inducer info
    # 0, 1, 2 for inducers blank, C, R; respectively
    columns = ['x','y','label','pad', 'dist']
    n_cols = len(columns)
    cor_pos_df = pd.DataFrame(np.empty((n_pos,n_cols)), columns=columns, index=np.arange(n_pos))
    for pos in np.arange(n_pos):
        label = pos_file_df.loc[pos,'fn'].split('/')[2]
        x, y = label_xy_dict[label]
        pad = pos_df.loc[pos,'pad']
        cor_pos_df.loc[pos,['x', 'y', 'label', 'pad']] = [x, y, label, pad]
    cor_pos_df.to_csv('corposdf.csv',index=False)

def plot_positions(cor_pos_df):
    plt.figure(figsize=(18,18))
    for p_i in np.arange(n_pos):
        colors = sns.color_palette('Set1', n_colors=n_pads)
        point_color = colors[np.int(cor_pos_df.loc[p_i,"pad"])]
        plt.plot(np.float(cor_pos_df.x[p_i]), 
                 -np.float(cor_pos_df.y[p_i]), 
                 '.', 
                 label=cor_pos_df.pad[p_i],
                 ms=20,
                 c=point_color)
        plt.text(np.float(cor_pos_df.x[p_i]), 
                 -np.float(cor_pos_df.y[p_i]), 
                 '{}'.format(p_i),
                 fontsize=14,
                 rotation=15)

    plt.gca().set_aspect('equal')
    plt.savefig('corposdf.png')
    plt.close('all')

um_pix = 1.6161

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

bg_tmpl = 'worker_outputs/bg_arr_{}_{}.npy'
# for chan_i, chan in enumerate([2,3]):
#     for frame_i in np.arange(n_frames):
#         bg_arr = get_bg_img(frame_i, chan)
#         np.save(bg_tmpl.format(frame_i, chan), bg_arr)

# out_fns = process_all(files_df, cor_pos_df)
# df = pd.concat([pd.read_csv(fn, index_col=None) for fn in out_fns])

