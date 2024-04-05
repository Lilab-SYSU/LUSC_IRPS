import os
import re
import sys

import numpy as np
import time
import argparse
import pdb
import pandas as pd
sys.path.append('/Data/yangml/Important_script/WSIimmune/WSI_Process')
from WSI_Process.WSIProcess import WSIProcess
from WSI_Process.WSI_function import StitchCoords,PatchImg

def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(255, 255, 255), alpha=-1, draw_grid=True)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_patch(WSIpath,save_dir,
              patch_size = 224, step_size = 224,
              seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                          'keep_ids': [], 'exclude_ids': []},
              filter_params={'a_t': 5, 'a_h': 10, 'max_n_holes': 5},
              vis_params={'vis_level': -1, 'line_thickness': 20},
              patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
              patch_level=0,
              use_default_params=False,
              seg=False, save_mask=True,
              stitch=False,
              patch=False,
              savepatchimg=False
              ):
    WSIprocessObj = WSIProcess(WSIpath)
    current_seg_params = seg_params.copy()
    current_vis_params = vis_params.copy()
    if current_vis_params['vis_level'] < 0:
        wsi = WSIprocessObj.getOpenSlide()
        if len(wsi.level_dimensions) == 1:
            current_vis_params['vis_level'] = 0

        else:
            best_level = wsi.get_best_level_for_downsample(64)
            current_vis_params['vis_level'] = best_level

    if current_seg_params['seg_level'] <0:
        wsi = WSIprocessObj.getOpenSlide()
        best_level = wsi.get_best_level_for_downsample(64)
        current_seg_params['seg_level'] = best_level

    path, _ = os.path.split(WSIpath)
    slide_id = WSIprocessObj.name
    current_patch_params = patch_params.copy()
    if seg:
        WSIprocessObj, seg_time_elapsed = segment(WSIprocessObj,seg_params=current_seg_params,filter_params=filter_params)
    if save_mask:
        mask = WSIprocessObj.visWSI(**current_vis_params)
        try:
            os.mkdir(save_dir+"/mask")
        except:
            print('Dirctory has existed.')
        save_mask_dir=save_dir+"/mask"
        mask_path = os.path.join(save_mask_dir, slide_id + '.jpg')
        mask.save(mask_path)

    if patch:
        try:
            os.mkdir(save_dir + "/patch")
        except:
            print('Dirctory has existed.')
        patch_save_dir=save_dir + "/patch"
        current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                     'save_path': patch_save_dir})
        file_path, patch_time_elapsed = patching(WSI_object=WSIprocessObj, **current_patch_params, )


    if stitch:
        try:
            os.mkdir(save_dir + "/stitch")
        except:
            print('Dirctory has existed.')
        stitch_save_dir=save_dir + "/stitch"
        file_path = os.path.join(patch_save_dir, slide_id + '.h5')
        if os.path.isfile(file_path):
            print('drawing heatmap!!!')
            heatmap, stitch_time_elapsed = stitching(file_path, WSIprocessObj, downscale=64)
            stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
            heatmap.save(stitch_path)
    if savepatchimg:
        try:
            os.makedirs(save_dir+"/patchimg/"+slide_id)
        except:
            print('Directory has existed.')
        patchimg_save_dir=save_dir+"/patchimg/"+slide_id
        file_path = os.path.join(patch_save_dir, slide_id + '.h5')
        PatchImg(file_path,patchimg_save_dir,WSIprocessObj)


parser = argparse.ArgumentParser(description='Slide segmentation and patching')
parser.add_argument('--WSIDir', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=224,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=224,
					help='patch_size')
parser.add_argument('--patch_level', type = int, default=0,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--savepatchimg', default=False, action='store_true')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
if __name__ == '__main__':
    args = parser.parse_args()
    files = os.listdir(args.WSIDir)
    slides=[slide for slide in files if re.search('.svs$',slide)]
    for slide in slides:
        WSIpath= os.path.join(args.WSIDir,slide)
        print(WSIpath)
        seg_patch(WSIpath=WSIpath,
                  patch_size=args.patch_size,
                  step_size=args.step_size,
                  patch_level=args.patch_level,
                  seg=args.seg,
                  save_dir=args.save_dir,
                  stitch=args.stitch,
                  patch=args.patch,
                  savepatchimg=args.savepatchimg
                  )

