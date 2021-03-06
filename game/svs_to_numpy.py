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
print('Please type the maximum number of svs files to be converted to numpy: ')
maximum_svs_files = input()
print('Please type the percentage threshold for 2.5x: ')
threshold_2_5 = input()
print('Please type the percentage threshold for 10x: ')
threshold_10= input()
if threshold_2_5 =="":
    threshold_2_5 = 0.2
else:
    threshold_2_5 = float(threshold_2_5)/100

if threshold_10 == "":
    threshold_10 = 0.75
else:
    threshold_10 = float(threshold_10) / 100



#input_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/"
#output_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/"

parser = argparse.ArgumentParser()

magnification = 2.5
tile_size = 512

tile_name_list =[]
completed_tile_name_list=[]
if path.isfile(output_path+"/Complete_tile_list.csv"):
    with open(output_path+"/Complete_tile_list.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                completed_tile_name_list.append(row['Tile_Name'])

for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith('.svs') and root+"/"+file not in completed_tile_name_list:
            tile_name_list.append(root+"/"+file)

def create_complete_list (path_csv, tile_name,Useful_Tile_Count,All_Tile_Count):
    if not path.isfile(path_csv):
        with open(path_csv, 'w') as csvfile:
            fieldnames = ['Tile_Name', 'Useful_Tile_Count','All_Tile_Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(path_csv, 'a', newline='') as csvfile:
        fieldnames = fieldnames = ['Tile_Name', 'Useful_Tile_Count','All_Tile_Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Tile_Name': tile_name, 'Useful_Tile_Count':Useful_Tile_Count, 'All_Tile_Count':All_Tile_Count})


def save_entry (path_csv, tile_name,tile_address_x,tile_address_y):
    if not path.isfile(path_csv):
        with open(path_csv, 'w') as csvfile:
            fieldnames = ['Tile_Name', 'Tile_address_x', 'Tile_address_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    useful_tiles=[]
    with open(path_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            useful_tiles.append([row['Tile_address_x'],row['Tile_address_y']])

    with open(path_csv, 'a', newline='') as csvfile:
        fieldnames = ['Tile_Name', 'Tile_address_x', 'Tile_address_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if [str(tile_address_x),str(tile_address_y)] not in useful_tiles:
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

def keep_tile_simple(tile,threshold):

    rgb = np.dot(tile[..., :3], [0.299, 0.587, 0.114])
    white_background = rgb > 215
    black_background = rgb < 40
    background = np.logical_or(white_background, black_background)
    pct = 1 - np.mean(background)
    if pct>threshold:
        return True
    else:
        return False



def get_tiles(tile_name,output_path):
    magnification = 2.5
    useful_tile_count=0
    all_tile_count=0
    useful_10x_simple=0
    useful_10x_complex=0
    slide = openslide.open_slide(tile_name) # loading the svs
    maximum_magnification = slide.properties['openslide.objective-power'] # finding maximum magnification available from svs
    generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0) # generate tiles from WSI

    # Find the level corresponding to 2.5x
    if magnification != int(slide.properties['openslide.objective-power']):
        level = generator.level_count - 1 - int(
            np.log2(np.float(slide.properties['openslide.objective-power']) / (magnification)))
    else:
        level = generator.level_count - 1
    [max_tile_address_x, max_tile_address_y] = np.array(generator.level_dimensions[level]) / tile_size
    for address_x in range(int(max_tile_address_x)):
        for address_y in range(int(max_tile_address_y)):
            all_tile_count+=1
            temp_tile = generator.get_tile(level, (address_x, address_y))
            bool_tile = keep_tile(np.asarray(temp_tile), tile_size, threshold_2_5)
            numpy_path = output_path+"/Svs_to_numpy/"+tile_name[tile_name.rfind("/")+1:-4]
            if bool_tile:
                if not os.path.exists(numpy_path):
                    os.makedirs(numpy_path)
                np.save(output_path+"/Svs_to_numpy/"+tile_name[tile_name.rfind("/")+1:-4]+"/Tile_x_"+str(address_x)+"_Tile_y_"+str(address_y),np.asarray(temp_tile))
                path_csv =numpy_path+"/useful_tile_list.csv"
                save_entry(path_csv, tile_name[tile_name.rfind("/")+1:-4], address_x, address_y)
                useful_tile_count+=1
                [[min_tile_address_x_10x, max_tile_address_x_10x], [min_tile_address_y_10x, max_tile_address_y_10x]] = [[4 * address_x, 4 * address_x + 4],[4 * address_y, 4 * address_y + 4]]
                x = np.arange(min_tile_address_x_10x, max_tile_address_x_10x, 1)
                y = np.arange(min_tile_address_y_10x, max_tile_address_y_10x, 1)
                for j in range(4):
                    for i in range(4):
                        temp_tile_10x = generator.get_tile(level+2, (x[i], y[j]))
                        bool_tile_10x = keep_tile_simple(np.asarray(temp_tile_10x), threshold_10)
                        if bool_tile_10x:
                            useful_10x_simple+=1
                            save_entry(numpy_path + "/useful_tile_list_10x.csv", tile_name[tile_name.rfind("/") + 1:-4], x[i], y[j])

    #print("Simple 10x: "+str(useful_10x_simple)+ " complex 10x: " +str(useful_10x_complex))
    return [all_tile_count,useful_tile_count]

for tile_i in range(min(len(tile_name_list),int(maximum_svs_files))):

    tile_name_list[tile_i][tile_name_list[tile_i].rfind("/")+1:-4] # get the name of svs file from the whole path
    t0 = time.time()
    [all_tile_count,useful_tile_count] = get_tiles(tile_name_list[tile_i],output_path)
    create_complete_list(output_path+"/Complete_tile_list.csv",tile_name_list[tile_i],useful_tile_count,all_tile_count)
    print("Slide " +str(tile_i+1)+ " took " +str(time.time() - t0)+" seconds with "+str(useful_tile_count)+ " useful and " +str(all_tile_count) +" total tiles.")