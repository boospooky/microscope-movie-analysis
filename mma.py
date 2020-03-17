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

def get_timevec_par(img_files):
    ctime_parallel_wrapper(img_files)
    time_vec = ctime_parallel_read_result(img_files)
    time_vec = time_vec - time_vec.min()
    return time_vec

def get_timevec(img_files):
    time_vec = np.array([ctime(fn) for fn in img_files])
    # time_vec = time_vec - time_vec.min()
    return time_vec

def get_exp_summary_from_fn(fn):
    all_metadata = fn_metadata_full(fn)
    summ_dict = all_metadata['Summary']
    chan_ind_list = [int(xx) for xx in summ_dict['ChNames']]
    chan_names = [all_channel_names[xx] for xx in chan_ind_list]
    n_chan = len(chan_ind_list)
    im_width = summ_dict['Width']
    im_height = summ_dict['Height']
    return n_chan, chan_ind_list, chan_names, im_width, im_height

def make_files_df(super_dir):
    img_files = find_tiffs_in_dir(super_dir)
    time_df_fn = os.path.join(super_dir, 'time_df.csv')
    # See if time_vec has been made previously
    if not os.path.isfile(time_df_fn):
        time_vec = get_timevec(img_files)
        time_df = pd.DataFrame(data={'time':time_vec,'fn':img_files}).sort_values(by='time')
        time_vec, img_files = time_df[['time','fn']].values.T
        time_df.to_csv(os.path.join(super_dir, 'time_df.csv'), index=False)
        print('printed {}'.format(time_df_fn))
    else:
        time_df = pd.read_csv(time_df_fn, index_col=None)
        time_vec, img_files = time_df[['time','fn']].values.T
    
    # Extract movie metadata from an image file
    if len(img_files) < 1:
        return 1
    out = get_exp_summary_from_fn(img_files[0])
    n_chan, chan_ind_lis, chan_names, im_width, im_height = out

    # Interpret pad, frame, channel from filenames
    img_re_pattern = r'{}/(?P<pad>[0-9]+)-Pos_(?P<padcol>[0-9]+)_(?P<padrow>[0-9]+)/.*_(?P<frame>[0-9]*)_(?P<channel>[0-9]*)_000.tif'.format(super_dir)
    rem = re.findall(img_re_pattern, '\n'.join(img_files), re.MULTILINE)
    rem_arr = np.array(rem)
    rem_arr[rem_arr==''] = -1
    metadata_arr = rem_arr.astype(np.int)
    n_rows = metadata_arr.shape[0]
    pad_vec, padcol_vec, padrow_vec, frame_vec, channel_vec = metadata_arr.T

    # Make dataframe
    # metadata and timevec data created by looping through img_files
    # therefore rows of each associated with the same file
    columns = ['pad','padcol','padrow','frame','channel','fn', 'time']
    col_data = [pad_vec, padcol_vec, padrow_vec, frame_vec, channel_vec, img_files, time_vec]
    files_df = pd.DataFrame(dict(zip(columns, col_data)))
    files_df.sort_values(by='time', inplace=True)
    
    # Each frame should be associated with the same number of images. 
    # If this is not the case, the acquisition was aborted.
    # Remove rows corresponding to the frames with too few images
    gb_frame = files_df.groupby('frame')
    imgs_per_frame = np.array([[key, len(sub_df)] for key, sub_df in gb_frame])
    max_ipf = np.max(imgs_per_frame[:,1])
    short_frames = imgs_per_frame[imgs_per_frame[:,1]>max_ipf,0]
    for frame_i in short_frames:
        drop_inds = gb_frame.get_group(frame_i).index
        files_df.drop(drop_inds, axis=0, inplace=True)
    files_df = files_df.reindex()
    files_df['pos'] = -np.ones(n_rows, dtype=np.int)
    gb_pos = files_df.groupby(['pad', 'padcol','padrow'])
    for i, inds in enumerate(gb_pos.groups.values()):
        files_df.loc[inds,'pos'] = i
    assert not np.any(files_df.pos == -1)
    return files_df

def make_positions_df(files_df):
    gb_frame = files_df.groupby('frame')

    n_pads = len(np.unique(files_df.pad))
    n_frames = np.max(files_df.frame)+1
    pos_df = files_df.loc[:,['pos','pad','fn']].groupby('pos').agg(np.min)
    n_pos = len(pos_df)

    all_metadata = fn_metadata_full(files_df['fn'].values[0])
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
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.savefig(out_fn)
    plt.close('all')

def write_movie_no_bg(out_fn, pad_ind, files_df, cor_pos_df, scale=4, skip=5, rotation=180, mag=10):
    # Get movie metadata
    n_chan, chan_ind_list, chan_names, im_width, im_height = get_exp_summary_from_fn(files_df.fn.values[0])
    img_width_options = [512,1024,2048]
    pixel_size_options = np.array([4,2,1])*1.6/(2.475*mag/10)
    zip_for_dict = zip(img_width_options,pixel_size_options)
    um_per_pixel_dict = dict(zip_for_dict)
    um_per_pixel = um_per_pixel_dict[im_width]
    pixels_per_um = 1/(um_per_pixel*scale)
    n_frames = files_df.frame.max()+1
    h, w = im_height//scale, im_width//scale
    uint_max = 65535
    # Setup plotting variables
    plt.close('all')
    pad_df = cor_pos_df.loc[cor_pos_df.pad==pad_ind,:]
    pos_list = np.unique(pad_df.index.values)
    xlims = np.array([pad_df.x.min(), pad_df.x.max()+1400])*pixels_per_um
    ylims = np.array([pad_df.y.min(), pad_df.y.max()+1400])*pixels_per_um
    rel_mins = np.concatenate([[xlims], [ylims]],axis=0).astype(np.int)
    pad_h = np.int(np.ceil(np.diff(ylims)))
    pad_w = np.int(np.ceil(np.diff(xlims)))
    fig_h, fig_w = np.array([pad_h, pad_w])/(100/scale)
    # Define helper functions
    def load_img(frame, pos, channel, rotation=rotation, scale=scale, sigma=3):
        frame_bool = files_df.frame==frame
        pos_bool = files_df.pos==pos
        chan_bool = files_df.channel == channel
        indx_bool = (frame_bool)&(pos_bool)&(chan_bool)
        if sum(indx_bool) < 1:
            print(pad_ind, frame, pos, channel)
            error_msg = 'Image not found when printing {}: pad:{} frame:{} pos:{} channel:{}'
            error_msg = error_msg.format(out_fn, pad_ind, frame, pos, channel)
            raise FileNotFoundError(error_msg)
        fn = files_df.loc[indx_bool,'fn'].values[0]
        img = skimage.filters.gaussian(skimage.io.imread(fn).astype(np.float), sigma)
        img = skimage.transform.downscale_local_mean(img, (scale, scale))
        return skimage.transform.rotate(img, rotation)

    pad_arr = np.zeros((pad_h, pad_w, n_chan))
    def get_frame_arr(frame_ind):
        bg_arr = np.zeros((h,w))
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
            frame_vals = pad_arr[:,:,chan_ind].flatten()
            frame_vals = np.sort(frame_vals)
            frame_n = len(frame_vals)
            ind_min, ind_max = np.array([0.05*frame_n, 0.95*frame_n],dtype=np.int)
            vmin, vmax = frame_vals[ind_min], frame_vals[ind_max]
            norm_fn = mpl_colors.Normalize(vmin, vmax, clip=True)
            norm_vec.append(norm_fn)
        for chan_ind, chan_slot in enumerate(chan_vec):
            norm = norm_vec[chan_ind]
            chan_arr = pad_arr[:,:,chan_ind:chan_ind+1]
            color_vec = mpl_colors.to_rgb(mpl_named_colors[chan_slot])
            img_arr += np.concatenate([norm(chan_arr)*color_val for color_val in color_vec],axis=2)
        im.set_array(img_arr)

    chan_vec = np.array(chan_ind_list)
    pos_lims = (pad_df[['x','y']]*pixels_per_um).astype(np.int)
    count_arr = np.zeros((pad_h, pad_w))
    for pos in pos_list:
        x0, y0 = (pos_lims.loc[pos,:].values - rel_mins[:,0]).astype(np.int)
        count_arr[y0:y0+h,x0:x0+w] += 1
    count_arr[count_arr<=0] = 1

    fig, ax = plt.subplots(1, 1, figsize=(fig_w,fig_h+1.2))
    img_arr = np.zeros((pad_h, pad_w, 3))
    im = ax.imshow(img_arr.copy(), animated=True, interpolation='none')
    #     cbar = fig.colorbar(im, ax=ax, ticks=[vmin, vmax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pad {}".format(pad_ind))

    fig.tight_layout()

    anim = anm.FuncAnimation(fig, animate, interval=100, frames=np.arange(0,n_frames,skip))
    plt.close('all')
    anim.save(out_fn, dpi=80, fps=5, writer='pillow')

class Acquisition():
    '''
    '''
    def __init__(self, in_str):
        if os.path.isdir(in_str):
            self._init_dir(in_str)
        elif os.path.isfile(in_str):
            self._init_file(in_str)
        else:
            raise(ValueError("Input string does not specify a directory or file"))
    
    def _init_dir(self, super_dir):
        self.super_dir = super_dir
        self.files_df = make_files_df(super_dir)
        self.expname = os.path.split(os.path.abspath(self.super_dir))[-1]
        if type(self.files_df) is not pd.DataFrame:
            if self.files_df == 1:
                print('error, aborting dir {}'.format(super_dir))
                return 1
        self.cor_pos_df = make_positions_df(self.files_df)
        metadata = get_exp_summary_from_fn(self.files_df['fn'].values[-1])
        self.n_chan, self.chan_ind_list, self.chan_names, self.im_width, self.im_height = metadata
        files_df_fn = os.path.abspath(os.path.join(super_dir, 'files_df.csv'))
        cor_pos_df_fn = os.path.abspath(os.path.join(super_dir, 'cor_pos_df.csv'))
        print('making csvs {} and {}'.format(files_df_fn, cor_pos_df_fn))
        self.files_df.to_csv(files_df_fn, index=False)
        self.cor_pos_df.to_csv(cor_pos_df_fn, index=False)
        for fn in [files_df_fn, cor_pos_df_fn]:
            assert os.path.isfile(fn)
        print('done writing {}'.format(super_dir))
    
    def _init_file(self, files_df_fn):
        self.super_dir = os.path.dirname(os.path.abspath(files_df_fn))
        self.expname = os.path.split(os.path.abspath(self.super_dir))[-1]
        self.files_df = pd.read_csv(files_df_fn, index_col=None)
        self.cor_pos_df = make_positions_df(self.files_df)
        metadata = get_exp_summary_from_fn(self.files_df['fn'].values[0])
        self.n_chan, self.chan_ind_list, self.chan_names, self.im_width, self.im_height = metadata

    def plot_positions(self, out_fn=None):
        if out_fn is None:
            out_fn = os.path.join(self.super_dir, '_cor_pos.png')
        plot_positions(self.cor_pos_df, out_fn)

    def write_movie_no_bg(self, pad_ind, scale=8, skip=1):
        out_fn = '{}_pad_{}.gif'.format(self.expname, pad_ind)
        out_fn = os.path.join(self.super_dir, out_fn)
        args = out_fn, pad_ind, self.files_df, self.cor_pos_df, scale, skip, self.rotation, self.mag
        try:
            write_movie_no_bg(*args)
        except FileNotFoundError as err:
            print(err)

    def write_all_pad_gifs_no_bg(self, scale=8, skip=1):
        pad_inds = np.unique(self.files_df.pad.values)
        for pad_ind in pad_inds:
            self.write_movie_no_bg(pad_ind, scale, skip)

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

# bg_tmpl = 'worker_outputs/bg_arr_{}_{}.npy'
# for chan_i, chan in enumerate([2,3]):
#     for frame_i in np.arange(n_frames):
#         bg_arr = get_bg_img(frame_i, chan)
#         np.save(bg_tmpl.format(frame_i, chan), bg_arr)

# out_fns = process_all(files_df, cor_pos_df)
# df = pd.concat([pd.read_csv(fn, index_col=None) for fn in out_fns])

with open('img_dirs.txt', 'r') as f:
    lines = f.read().splitlines()

with Pool(4) as pool:
    res = pool.map_async(Acquisition, lines)
    res.wait()
    pool.close()
    pool.join()
    acq_list = res.get()

for acq in acq_list:
    acq.rotation = 180
    acq.mag = 10
    acq.write_all_pad_gifs_no_bg()
