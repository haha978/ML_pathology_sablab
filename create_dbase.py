import h5py
import torch
import os
import numpy as np

def main():
    #need to create h5py database
    dbase = '/home/jm2239/model/slides_dbase.hdf5'
    f = h5py.File( dbase , "w")
    dict_dir = '/home/jm2239/gliomadset/'
    dict_paths = [dict_dir + file for file in os.listdir(dict_dir) if file[-3:] == '.pt']
    print("Adding "+ str(len(dict_paths))+" slides to the created database.")
    print(dict_paths)
    f = h5py.File(dbase , "a")
    for path in dict_paths:
        add_dict(f,path)
    f.close()

def add_dict(file, path):
    """
    add a dictionary to the database(file). Dictionary is stored in given path.
    """
    lib = torch.load(path)
    _, slide_name = os.path.split(lib['slides'])
    slide_num = slide_name[:-4]
    grp = file.create_group(slide_num)
    grp.create_dataset("tiles", data = lib["tiles"])
    grp.create_dataset("grids", data = lib["grids"])
    grp.create_dataset("targets", data = lib["targets"])
    grp.create_dataset("level", data = lib["level"])
    print("Finished adding slide: " + slide_name + " to the database")

if __name__ == '__main__':
    main()
