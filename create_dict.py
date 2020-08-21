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

parser = argparse.ArgumentParser(description = 'Preprocessing images for tumor classification')
parser.add_argument('--input_dir', type=str, default='.', help='name of directory that contains input WSIs. Need "/" at the end')
'''
creates dictionary for MIL training
'''
def main():
    global args
    args = parser.parse_args()
    IA_IDH_WT,IA_IDH_MUTANT = slide_list(args.input_dir)

    slides1 = IA_IDH_WT
    slides2 = IA_IDH_MUTANT
    targets1 = [0]*len(IA_IDH_WT)
    targets2 = [1]*len(IA_IDH_MUTANT)
    mult1 = [1.]*len(slides1)
    mult2 = [1.]*len(slides2)
    slides = slides1 + slides2
    targets = targets1 + targets2
    mult = mult1 + mult2
    slides_data = list(zip(slides,targets,mult))
    print(slides_data)
    print("Start processing "+str(len(slides_data))+" slides")
    pool = ThreadPool()
    pool.map(slide_dict,slides_data)
    pool.close()
    pool.join()

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
    level, tiles = level_and_tiles(t_list)
    dict = {'slides':slide_path,'tiles':tiles,'targets':target,'mult':mult,'level':level}
    save_dir ='/home/jm2239/sets/'
    _, file_name = os.path.split(slide_path)
    torch.save(dict, save_dir + file_name[:-4] + '_dict.pt')

def level_and_tiles(t_list):
    """
    Return (level, tiles) tuples where
    level: WSI pyramid level (integer) from which to read the tiles.
    tiles: list of list of tiles, where maximum of 100 tiles are from each WSIs.
    Choose 100 tiles randomly if there are more than 100 tiles per slide
    IF not, choose all tiles available for each slide

    t_list is a list of tuples. First element of a tuple indicates level
    and the second element of a tuple is list of tiles
    """
    level, all_tiles = t_list
    tiles = choose_tiles(all_tiles)

    return level,tiles

def choose_tiles(tile_list):
    """
    Returns list of tiles from tile_list.
    Choose 100 tiles randomly if there are more than 100 tiles in tile_list
    IF not, choose all tiles available in tile_list
    """
    tiles = []
    all_tiles_len = len(tile_list)
    len_tiles = min(all_tiles_len,100)
    #create random sequence of integer from range(len_tiles) without replacement
    idxs = random.sample(range(all_tiles_len),len_tiles)
    for idx in idxs:
        tiles.append(tile_list[idx])
    return tiles

def slide_list(input_dir):
    """
    slide_list(input_dir) returns a tuple that contains lists of certain slide path.
    The first element contains slide paths of slides with IA_IDH_WT labels.
    The second element contains slide paths of slides with IA_IDH_MUTANT labels.
    input_dir is a directory that contains all WSIs. input_dir ends with /.
    """
    #need to check whether the slide has target as IA_IDH_WT or IA_IDH_MUTANT
    csv = pd.read_csv('SCAN_MASTER_DEID_6_1_2020.csv')
    IA_IDH_WT_csv = csv.loc[csv["TUMOR_CLASS"]=="IA_IDH_WT", "IMAGE_ID"]
    IA_IDH_MUTANT_csv = csv.loc[csv["TUMOR_CLASS"]=="IA_IDH_MUTANT","IMAGE_ID"]
    # list of IA_IDH_WT, IA_IDH_MUTANT IMAGE_ID list
    IA_IDH_WT_csv = IA_IDH_WT_csv.tolist()
    IA_IDH_MUTANT_csv = IA_IDH_MUTANT_csv.tolist()
    IA_IDH_WT_csv = [str(x) for x in IA_IDH_WT_csv]
    IA_IDH_MUTANT_csv = [str(x) for x in IA_IDH_MUTANT_csv]
    #save WSIs that are actually in input_dir
    IA_IDH_WT = []
    IA_IDH_MUTANT = []
    for file_name in os.listdir(input_dir):
        if (file_name[-4:] == '.svs'):
            if file_name[:-4] in IA_IDH_WT_csv:
                IA_IDH_WT.append(input_dir + file_name)
            elif file_name[:-4] in IA_IDH_MUTANT_csv:
                IA_IDH_MUTANT.append(input_dir + file_name)
    return (IA_IDH_WT,IA_IDH_MUTANT)

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
    Returns level and appropriate tiles for given slide_path
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
    #attain 10x
    level = level - 1
    cols, rows = generator.level_tiles[level]
    #224: tile_size, 0: overlap

    kept_tiles = []
    for col in range(cols):
        print('Finished processing '+ str(col) +'/'+str(cols)+' of slide: '+file_name)
        for row in range(rows):
            tile = np.asarray(generator.get_tile(level, (col,row)))
            if keep_tile(tile, tile.shape[0], 0.75):
                kept_tiles.append(tile)
    end_time = time.time()
    diff_time = end_time - start_time
    print('Time took for processing slide ' + file_name +': ',diff_time)
    start_time = time.time()
    save_tiles(kept_tiles,unkept_tiles,file_name[:-4])
    end_time = time.time()
    diff_time = start_time - end_time
    print('Time took to save random tiles from slide ' + file_name + str(diff_time))
    return (level,kept_tiles)

def save_tiles(kept,unkept,slide_num):
    """
    Save some tiles that passed / not passed the keep_tile() function
    """
    tiles_dir = '/home/jm2239/tiles_jpg/' + slide_num + '/'
    os.mkdir(tiles_dir)
    kept_tiles_dir = tiles_dir+'kept_tiles/'
    os.mkdir(kept_tiles_dir)
    unkept_tiles_dir = tiles_dir+'unkept_tiles/'
    os.mkdir(unkept_tiles_dir)
    kept_save = random.sample(kept,min(len(kept),100))
    for i, tile in enumerate(kept_save):
        im = Image.fromarray(tile)
        im.save(kept_tiles_dir+str(i)+'.jpg')
    unkept_save = random.sample(unkept,min(len(unkept),100))
    for i, tile in enumerate(unkept_save):
        im = Image.fromarray(tile)
        im.save(unkept_tiles_dir+str(i)+'.jpg')

def split_data(IA_IDH_WT_list):
    """
    Returns two lists that contain paths to slide with IA_IDH_WT labels
    """
    dict_dir = '/home/jm2239/sets/'
    dict_dir_list = os.listdir(dict_dir)
    dict_num_list = []
    for dict in dict_dir_list:
        dict_num_list.append(dict[:6])
    excluded_l = []
    for path in IA_IDH_WT_list:
        _ , filename = os.path.split(path)
        if filename[:-4] not in dict_num_list:
            excluded_l.append(path)
    third = int(len(excluded_l)/3)
    l1 = excluded_l[:third]
    l2 = excluded_l[third:]
    return l1,l2


if __name__ == '__main__':
    main()
