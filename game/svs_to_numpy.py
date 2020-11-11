import os
import random
import time
import argparse
import csv
import numpy as np
import openslide
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import pathlib
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from os import path
import time

print('Please type input path: ')
input_path = input()
print('Please type the output path: ')
output_path = input()

#input_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/"
#output_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/"

parser = argparse.ArgumentParser()

magnification = 2.5
tile_size = 512

tile_name_list =[]

for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith('.svs') and root+"/"+file:
            tile_name_list.append(root+"/"+file)

def save_entry (path_csv, tile_name,tile_address_x,tile_address_y):
    if not path.isfile(path_csv):
        with open(path_csv, 'w') as csvfile:
            fieldnames = ['Tile_Name', 'Tile_address_x', 'Tile_address_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(path_csv, 'a', newline='') as csvfile:
        fieldnames = ['Tile_Name', 'Tile_address_x', 'Tile_address_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Tile_Name': tile_name, 'Tile_address_x':tile_address_x, 'Tile_address_y':tile_address_y})

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


def get_tiles(tile_name,output_path):
    magnification = 2.5
    useful_tile_count=0
    slide = openslide.open_slide(tile_name) # loading the svs
    maximum_magnification = slide.properties['openslide.objective-power'] # finding maximum magnification available from svs
    generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0) # generate tiles from WSI

    # Find the level corresponding to 2.5x
    if magnification != int(slide.properties['openslide.objective-power']):
        level = generator.level_count - 1 - int(
            np.sqrt(np.float(slide.properties['openslide.objective-power']) / (magnification)))
    else:
        level = generator.level_count - 1
    [max_tile_address_x, max_tile_address_y] = np.array(generator.level_dimensions[level]) / tile_size
    for address_x in range(int(max_tile_address_x)):
        for address_y in range(int(max_tile_address_y)):
            temp_tile = generator.get_tile(level, (address_x, address_y))
            bool_tile = keep_tile(np.asarray(temp_tile), tile_size, 0.75)
            numpy_path = output_path+"/Svs_to_numpy/"+tile_name[tile_name.rfind("/")+1:-4]
            if bool_tile:
                if not os.path.exists(numpy_path):
                    os.makedirs(numpy_path)
                np.save(output_path+"/Svs_to_numpy/"+tile_name[tile_name.rfind("/")+1:-4]+"/Tile_x_"+str(address_x)+"_Tile_y_"+str(address_y),np.asarray(temp_tile))
                path_csv =numpy_path+"/useful_tile_list.csv"
                save_entry(path_csv, tile_name[tile_name.rfind("/")+1:-4], address_x, address_y)
                useful_tile_count+=1
    return useful_tile_count

for i in range(len(tile_name_list)):
    tile_name_list[i][tile_name_list[i].rfind("/")+1:-4] # get the name of svs file from the whole path
    t0 = time.time()
    useful_tile_count = get_tiles(tile_name_list[i],output_path)
    print("Slide " +str(i+1)+ " took " +str(time.time() - t0)+" seconds with "+str(useful_tile_count)+ " tiles.")