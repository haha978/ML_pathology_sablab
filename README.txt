Written date: 08/31/2020

- List of input directories that contains raw Whole Slide Images(WSIs):
1. TCGA WSIs: '/nfs02/data/TCGA/TCGA_brain/'
2. pathologists' WSIs: '/nfs03/data/weill_pathology/cajalnet_data/'

- List of directories that contains slide dictionaries from WSIs:
The data in the slide dictionaries are used to create databases(hdf5 file)
Such additional step was taken to incorporate multi-threading easily.

1. '/home/jm2239/gliomadset/10x/' stores slide dictionaries, which contains
    10x preprocessed tiles from pathologists WSIs.
2. '/home/jm2239/gliomadset/2.5x/'stores slide dictionaries, which contains
    2.5x preprocessed tiles from pathologists WSIs.
3. '/home/jm2239/TCGA_data/2.5x/' stores slide dictionaries, which contains
    2.5x preprocessed tiles from TCGA_brain data

-List of databases created from WSIs:
1. '/home/jm2239/TCGA_data/TCGA_2_5x_dbase.hdf5' stores 2.5x preprocessed tiles
    from TCGA_brain WSIs.
2. '/home/jm2239/gliomadset/slides_2_5x.hdf5' stores 2.5x preprocessed tiles
    from pathologists' WSIs.
3. '/home/jm2239/gliomadset/slides_10x.hdf5' stores 10x preprocessed tiles from
    pathologists' WSIs

Structure of database can inferred from 'create_dbase.py'

-How to create database:

1. Use 'create_dict.py' to create slide dictionaries from WSIs. Need to
change the input/output directories appropriately. slide dictionaries store
preprocessed tiles and their data
2. Use 'create_dbase.py' to create a database(hdf5 file) from stored slide dictionaries.
Need to change the input/output directories appropriately.
