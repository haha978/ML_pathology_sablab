import h5py
import numpy as np
"""
Add augmented tiles to the database
"""

def augment_tiles(tiles):
    # Returns tiles that are 90, 180, 270 degrees rotated and tiles that
    # are mirrored
    augmented_tiles = []
    for idx in range(len(tiles)):
        im = tiles[idx]
        im_fliplr = np.fliplr(im)
        im_flipud = np.flipud(im)
        im_flip = np.flip(im)
        im_90 = np.rot90(im)
        im_180 = np.rot90(im_90)
        im_270 = np.rot90(im_180)
        augmented_tiles.extend([im_fliplr,im_flipud,im_flip,im_90,im_180,im_270])
    print("Created " + str(len(augmented_tiles)) +" augmented tiles from "+ str(len(tiles)) + " original tiles")
    return augmented_tiles

def add_augmented_tiles(dbase_path):
    #add augmented tiles to the database
    dbase = h5py.File(dbase_path,'a')
    for key in dbase.keys():
        grp = dbase[key]
        augmented_tiles = augment_tiles(grp['tiles'][:])
        grp.create_dataset("augmented tiles", data = augmented_tiles)
        print("Finished adding augmented tiles from "+ str(key) + " to the database")
    dbase.close()

def main():
    dbase_path = '/home/jm2239/TCGA_data/TCGA_2_5x_dbase.hdf5'
    add_augmented_tiles(dbase_path)

if __name__ == '__main__':
    main()
