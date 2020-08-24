import math
import time
import os
import pandas as pd
import torch
import numpy as np
import openslide
from PIL import Image
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
import argparse
import random
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def get_row_index(csv):
    row_indexs = []
    for row in csv.itertuples():
        if row[6] not in row_indexs:
            row_indexs.append(row[6])
    return row_indexs

def slide_list(input_dir):
    """
    Returns list of nonglioma slides from input_dir 
    """
    csv = pd.read_csv('/home/jm2239/SCAN_MASTER_DEID_6_1_2020.csv')
    indexs = get_row_index(csv)
    #these are infiltrating gliomas
    indexs.remove("IA_IDH_WT")
    indexs.remove("IA_IDH_MUTANT")
    indexs.remove("IA_IDH_WT_PEDIATRIC")
    indexs.remove("IA_IDH_UNKNOWN")
    indexs.remove("OLIGO")
    nonglioma_list = []
    for label in indexs:
        label_csv = csv.loc[csv["TUMOR_CLASS"]==label, "IMAGE_ID"]
        label_list = label_csv.tolist()
        label_list = [str(x) for x in label_list]
        print(label)
        print(label_list)
        nonglioma_list.extend(label_list)
    nonglioma_paths = []
    for file_name in os.listdir(input_dir):
        if (file_name[-4:] == '.svs'):
            if file_name[:-4] in nonglioma_list:
                nonglioma_paths.append(input_dir+file_name)
    return nonglioma_paths

def optical_density(tile):
    """
    Convert a tile to optical density values.

    Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).

    Returns:
    A 3D NumPy array of shape (tile_size, tile_size, channels)
    representing optical density values.
    """
    tile = tile.astype(np.float64)
    od = -np.log((tile + 1) / 240)
    return od

def keep_tile(tile, tile_size, tissue_threshold):
    """
    Determine if a tile should be kept.

    This filters out tiles based on size and a tissue percentage
    threshold, using a custom algorithm. If a tile has height &
    width equal to (tile_size, tile_size), and contains greater
    than or equal to the given percentage, then it will be kept;
    otherwise it will be filtered out.

    Args:
     tile: tile is a 3D NumPy array of shape (tile_size, tile_size, channels).
     tile_size: The width and height of a square tile to be generated.
     tissue_threshold: Tissue percentage threshold.

    Returns:
     A Boolean indicating whether or not a tile should be kept for
     future usage.
    """
    if tile.shape[0:2] == (tile_size, tile_size):
        tile_orig = tile

        # Check 1
        # Convert 3D RGB image to 2D grayscale image, from
        # 0 (dense tissue) to 1 (plain background).
        tile = rgb2gray(tile)
        # 8-bit depth complement, from 1 (dense tissue)
        # to 0 (plain background).
        tile = 1 - tile
        # # Canny edge detection with hysteresis thresholding.
        # # This returns a binary map of edges, with 1 equal to
        # # an edge. The idea is that tissue would be full of
        # # edges, while background would not.
        tile = canny(tile)
        # Binary closing, which is a dilation followed by
        # an erosion. This removes small dark spots, which
        # helps remove noise in the background.
        tile = binary_closing(tile, disk(10))
        # Binary dilation, which enlarges bright areas,
        # and shrinks dark areas. This helps fill in holes
        # within regions of tissue.
        tile = binary_dilation(tile, disk(10))
        # Fill remaining holes within regions of tissue.
        tile = binary_fill_holes(tile)
        # Calculate percentage of tissue coverage.
        percentage = tile.mean()
        check1 = percentage >= tissue_threshold

        # Check 2
        # Convert to optical density values
        tile = optical_density(tile_orig)
        # Threshold at beta
        beta = 0.15
        tile = np.min(tile, axis=2) >= beta
        # Apply morphology for same reasons as above.
        tile = binary_closing(tile, disk(2))
        tile = binary_dilation(tile, disk(2))
        percentage = tile.mean()
        check2 = percentage >= tissue_threshold
        #set the tile back to the original tile
        tile = tile_orig
        return check1 and check2

    else:
        return False

def process_svs(slide_path):
    """
    Returns level, grids and appropriate tiles for given slide_path
    """
    start_time = time.time()
    #read the slides
    slide = openslide.open_slide(slide_path)
    #split filepath and filename
    filepath, file_name = os.path.split(slide_path)
    generator = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
    highest_zoom_level = generator.level_count - 1
    try:
        #slide is NOT GENERATOR but the given .svs file
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        # need the offset as some .svs files can be either 20X or 40x
        # objective goes down x2 if level decreases by 1
        offset = math.floor((mag / 20) / 2)
        level = highest_zoom_level - offset
    except (ValueError, KeyError) as e:
        level = highest_zoom_level
    #attain 2.5x
    level = level - 3
    cols, rows = generator.level_tiles[level]
    #224: tile_size, 0: overlap

    kept_tiles = []
    kept_grids = []
    for col in range(cols):
        print('Finished processing '+ str(col) +'/'+str(cols)+' of slide: '+file_name)
        for row in range(rows):
            tile = np.asarray(generator.get_tile(level, (col,row)))
            if keep_tile(tile, tile.shape[0], 0.75):
                kept_tiles.append(tile)
                kept_grids.append((col,row))
    end_time = time.time()
    diff_time = end_time - start_time
    print('Time took for processing slide ' + file_name +': ',diff_time)
    return (level,kept_tiles,kept_grids)

def slide_dict(slide_data):I
    """
    Saves a dictionary of one WSI, once the processing is done.
    Dictionary has keys 'slides','tiles','targets','mult', and 'level'
    <PARAMETERS>
    slide_data: (slide path, target, mult) triplet
    """
    slide_path, target, mult = slide_data
    #t_list contains level and tiles for given slide's slide_path
    t_list = process_svs(slide_path)
    level, tiles, grids = t_list
    dict = {'slides':slide_path,'tiles':tiles,'grids':grids, 'targets':target,'mult':mult,'level':level}
    save_dir ='/home/jm2239/gliomadset/2.5x/'
    _, file_name = os.path.split(slide_path)
    torch.save(dict, save_dir + file_name[:-4] + '_dict.pt')

def split_half(list1):
    n = len(list1)
    half = n //2
    return list1[:half], list1[half:]

def main():
    non_glioma_paths = slide_list("/nfs03/data/weill_pathology/cajalnet_data/")
    slides1, slides2= split_half(non_glioma_paths)
    #nonglioma = label with 0
    targets1 = [0]*len(slides1)
    targets2 = [0]*len(slides2)
    mult1 = [1.]*len(slides1)
    mult2 = [1.]*len(slides2)
    slides = slides1 + slides2
    print("The number of full slidepaths: ", len(slides))
    print("The number of processing slidepaths: ", len(slides2))
    targets = targets1 + targets2
    mult = mult1 + mult2
    slides_data = list(zip(slides2,targets2,mult2))
    print(slides_data)
    print("Start processing "+str(len(slides_data))+" slides")
    pool = ThreadPool()
    pool.map(slide_dict,slides_data)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
