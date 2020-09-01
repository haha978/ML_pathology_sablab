# this is a library of functions to generate a preprocessing pipeline for whole slide scans

import openslide, PIL, os, sys, time, math, random, keras, csv, json, aggdraw, traceback, warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from iteration_utilities import deepflatten
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras import backend as K

# since everything seems to be working, and this error started coming up after I updated some packages,
# I've been unable to track down the source of the warning, so I finally decided to just turn it off
pd.set_option("mode.chained_assignment", None)

# warning tracking
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

# a few useful constants
sqrt2 = math.sqrt(2)
sqrt2min1 = sqrt2 - 1.0
get_depth = lambda L: isinstance(L, list) and max(map(get_depth, L))+1

# this is a function which takes a text file and returns 
# a list of informative tile coordinates
def generate_patch_list(source_txt, verbose = False):
    patch_file = open(source_txt, "r")
    patch_list = []
    current_line =  patch_file.readline()
    current_line = patch_file.readline()[0:-1]
    while current_line != '':
        line_coordinates = str.split(current_line, " ")
        try:
            patch_list.append((int(line_coordinates[0]),
                           int(line_coordinates[1]),
                           int(line_coordinates[2]),
                           float(line_coordinates[3])))
            current_line = patch_file.readline()[0:-1]
        except: 
            print(f"Error reading {source_txt} {line_coordinates}")
            print(f"Line after error = {current_line}")
    return patch_list

# this is a pair of functions that takes a whole slide scan as an input 'input_wss'
# and generates a txt file which finds tiles of size 'patch_size' at all scan levels
# which have sufficient internal variability to be considered informative
# the threshhold value is the cutoff standard deviation of pixel intesities
# above which a patch is considered informative.  The main function to call for any given
# whole slide scan file is the whitespace_remover function, which creates the necessary
# text files and makes sure the slide exists, and then calls interrogate_subtiles across
# each patch, starting at the lowest level of downsampling.

# interrogate subtiles then determines if the patch should be considered informative,
# based on the absence of whitespace and variability of pixel intensity.  The coordinates of 
# informative tiles are then recorded in the output_txt.  The informative tile is then 
# subtiled at the next deeper level.  This top-down approach should reduce the
# number of times uninformative regions are re-sampled.  To facilitate counting,
# this function returns 1 if a tile is generated, and 0 if no tile is generated

# the output file is of the format:
# patch_size = XXX           (appears only once at the start of the file)
# followed by a very long list of tile coordinates (1 tile per line)
# of the format [level] [x_corner] [y_corner] [level_downsample]
# where (x_start, y_start) is the coordinate of the pixel of the tile relative to level 0, 
# which is useful for the openslide.read_region() function
#
# these rows could then be used to recall the openslide file as follows:
# patch = [openslide_object].read_region((patch_tuple[0],patch_tuple[1]),patch_tuple[2],(patch_size,patch_size))
# 
# Non-overlapping tiles are generated.  These can then be fed into a generator, and
# overlappingg tiles could easilybe generated just by randomly shifting the x,y coordinates
# by random.randint(-(patch_size // 2), patch_size // 2).  Rotation by an arbitrary 
# (non-90 degree) angle would also be possible.
# A less important consideration, because tiles can be pulled directly from the .svs,
# less storage space is needed to store patches

def interrogate_subpatches(scan : openslide, output_txt_writer, level,
                         starting_x, starting_y,
                         patch_size = 512, white_threshold = 690, 
                         percent_threshold = 25,  verbose = True):
    # if the current level is the deepest level, there's no need to continue
    if level < 0:
        return 0
    
    # it's a good idea to be more permissive in large tiles
    percent_threshold_for_level = percent_threshold / ((level + 1) ** 2)
    
    # read patch from scan
    patch = tf.constant(image.img_to_array(scan.read_region((starting_x, starting_y), 
                     level, 
                     (int(patch_size), 
                      int(patch_size)))))
    # if a pixel has alpha == 0, it is considered uninformative
    patch_alpha_not_0 = tf.not_equal(patch[:,:,3],0)

    # if a RGB sum is over a set intensity, pixel in considered uninformative
    patch_non_whitespace = tf.less_equal(
        tf.reduce_sum(patch[:,:,0:3], axis = 2), white_threshold)

    # percent informative pixels = sum of informative pixels / patch_size ** 2
    patch_percent_informative = tf.reduce_sum(
        tf.cast(tf.logical_and(patch_alpha_not_0, 
                               patch_non_whitespace), 
                tf.float32)) * 100. / (patch_size ** 2)

    # if patch has sufficient informative pixels, write to output_txt,
    # then iterate through subtiles
    if percent_threshold_for_level <= patch_percent_informative:
        output_txt_writer.write(f"{level} {int(starting_x)} {int(starting_y)} {float(scan.level_downsamples[level])}\n")
        del patch
        del patch_alpha_not_0
        del patch_non_whitespace
        del patch_percent_informative
        if verbose:
            sys.stdout.write(f"level {level} {starting_x} {starting_y}              \r") 
        if level > 0:
            for i in tf.range(starting_x, starting_x + 
                              int(patch_size * 
                                  scan.level_downsamples[level]),
                              int(patch_size * 
                                  int(scan.level_downsamples[level - 1]))):
                for j in tf.range(starting_y, starting_y + 
                                  int(patch_size * 
                                      scan.level_downsamples[level]),
                                  int(patch_size * 
                                      int(scan.level_downsamples[level - 1]))):
                        interrogate_subpatches(scan, output_txt_writer, level - 1,
                                  i, j, patch_size, white_threshold, 
                                  percent_threshold, verbose)
    return
                        
def informative_patch_finder(input_wss, input_dir, output_txt = None, 
                output_dir = None, patch_size = 512, verbose = True, 
                overwrite = False, white_threshold = 690, 
                percent_threshold = 50,  target_memory_size = 8e8):
    
  # If output_path or output_txt is not named, set output_path to input_path.
  # If the output_txt filename and output_path are not specified, we generate a filename of the format:
    # [svsfile file prefix]_[patch_size].txt in the same directory as input_wss.
    # Regardless, the two need to be os.path.joined()
    #
    # It also returns the list itself
    
    if output_dir == None:
        output_dir = input_dir
    if output_txt == None:
        output_txt = os.path.join(output_dir, str.split(input_wss,".")[0] + "_" + str(patch_size) + ".txt")
    else:
        output_txt = os.path.join(output_dir,output_txt)

    # open our wss scan file in openslide
    # if slide is not valid, exit with an error message
    try:
        scan = openslide.OpenSlide(os.path.join(input_dir,input_wss))
    except:
        print("Failed to open ",input_wss)
        return
    
    # if the output_txt file exists, and overwrite flag == False, return the list
    if os.path.isfile(output_txt):
        if overwrite == False:
            return list(generate_patch_list(output_txt, verbose = verbose))
        
    print("Processing " + input_wss + " to " + output_txt)
    
    # get level dimensions
    i_size = scan.level_dimensions[0,0]
    j_size = scan.level_dimensions[0,1]

    # create output file.  
    outfile = open(output_txt, mode = 'w')
    outfile.write("patch_size = " + str(patch_size) + "\n")

    # scan through the entire slide at each level, and write to outfile accordingly
    start_time = time.time()
    for i in range(0, i_size,
                   int(scan.level_downsamples[scan.level_count - 1]) * 
                   patch_size):
        for j in range(0, j_size,
                  int(scan.level_downsamples[scan.level_count - 1]) * 
                  patch_size):
            interrogate_subpatches(scan, outfile, scan.level_count - 1,
                     i, j,
                     patch_size = patch_size, white_threshold = white_threshold, 
                     percent_threshold = percent_threshold,
                     verbose = verbose)     
    #close to clean up
    outfile.close()
    if verbose:
        sys.stdout.write(input_wss + f' complete! Area:{scan.level_dimensions[0][0]} x {scan.level_dimensions[0][1]}, time elapsed {time.time() - start_time:.2f} seconds       \n')
    scan.close()
    return list(generate_patch_list(output_txt, verbose = verbose))

#### this is a function that takes as an input an openslide file (.svs or other supported, 
#### level, starting x, starting y, level_downsample factor, patch_size, and 
#### an arbitrary theta (random if None), and generates an array corresponding to the 
#### augmented image tile, with an arbitrary random translation within the range +/- 
#### (patch_size * level_downsamples) // 2, rotated to theta degrees counterclockwise.
#### rgb factors simulate variability in staining between labs
#### The interpolation method can be specified, but defaults to 2 (BILINEAR)

# to keep the code simple, the largest possible area needed for a rotation is when theta
# is 45 degrees, in which case the largest axis will be equal to 
# sqrt(2) * patchsize, so we can sample a region of that size, with some slight offsetting

def generate_augmented_tile(input_wss : openslide, coordinate_list, patch_size,
                            x_shift = 0, y_shift = 0,
                           r_factor = 1.0, g_factor = 1.0, b_factor = 1.0,
                             theta = 0, transposition = 0,
                             interpolation = 2):
    level, x0, y0, downsample = coordinate_list
    
    scan = openslide.open_slide(input_wss)

    x0 += x_shift
    y0 += y_shift
        
    # pad the area to be sampled for rotation
    x0_expanded = int(x0 - (downsample * patch_size * sqrt2min1) / 2)
    y0_expanded = int(y0 - (downsample * patch_size * sqrt2min1) / 2)
    
    # read the padded region, rotate it by theta (around center), 
    # and crop back to patch_size
    rotated_tile = scan.read_region((x0_expanded, y0_expanded),
        level, (int(patch_size * sqrt2), int(patch_size * sqrt2))).rotate(theta, 
            resample = interpolation, expand = True)
        
    augmented_tile = image.img_to_array(rotated_tile.crop(
                    ((rotated_tile.size[0] - patch_size) // 2,
                     (rotated_tile.size[1] - patch_size) // 2,
                     (rotated_tile.size[0] + patch_size) // 2,
                     (rotated_tile.size[1] + patch_size) // 2)))
    
    # switch the alpha channel such that non-informative is 1, informative is 0
    augmented_tile[:,:,3] = np.logical_not(augmented_tile[:,:,3])
    
    # set tiles that are out of the window to be approximately whitespace
    augmented_tile[:,:,0] += 242 * augmented_tile[:,:,3]
    augmented_tile[:,:,1] += 242 * augmented_tile[:,:,3]
    augmented_tile[:,:,2] += 242 * augmented_tile[:,:,3]

    # to simulate variability in staining between labs, slightly vary the RGB channels
    augmented_tile[:,:,0] *= r_factor
    augmented_tile[:,:,1] *= g_factor
    augmented_tile[:,:,2] *= b_factor
    
    if transposition:
        return np.transpose(augmented_tile[:,:,0:3], axes = (1,0,2)) / 255.0
    
    # return the rgb channels of the augmented tile
    return augmented_tile[:,:,0:3] / 255.0

# generates a tensor corresponding to manually annotated regions
# 
# annotation_classes
#    a list of dictionaries corresponding to the names of features to draw
# coordinate_list
#    a tuple of coordinates corresponding to level, x0, y0, and downsampling factor
# patch_size
#    integer value, number of pixels in width and height for a square tile
# segmentation_output_size
#    the size (in pixels) of the output_tensor
# missing_mode dictates how to handle entries without a segmentation file
#    possible modes "zeros", "ones", "random", and "average"
#    defaults to "random"
# image_id is included only for the purpose of error tracking, and has no other function
def generate_augmented_segmentation_tile(annotation_classes, 
                                         coordinate_list,
                                         annotation_list = None,
                                         missing_mode = "random",
                                         patch_size= 512, 
                                         segmentation_output_size = None, 
                                         x_shift = 0, y_shift = 0,
                                         theta = 0, 
                                         transposition = None, 
                                         interpolation = 2,
                                         image_id = None,
                                         verbose = False):
  
  # generate an output tensor of segmentation_output_size x segmentation_output_size x len(annotation_classes)
  # If no annotation file exists, set the entire tensor to a median value (1.0 / len(annotation_classes))
  
  if segmentation_output_size == None:
    segmentation_output_size = patch_size

  # handle missing entries as determined by "missing mode"
  if annotation_list == None:
    if missing_mode == "ones":
      return np.ones(shape = (segmentation_output_size,
                          segmentation_output_size,
                          len(annotation_classes)),
                          dtype = np.float32)
    if missing_mode == "zeros":
      return np.zeros(shape = (segmentation_output_size,
                          segmentation_output_size,
                          len(annotation_classes)),
                          dtype = np.float32)
    if  missing_mode == "average":
      return np.ones(shape = (segmentation_output_size,
                          segmentation_output_size,
                          len(annotation_classes)),
                          dtype = np.float32) / len(annotation_classes)
    # if missing, and mode not specified, defaults to random
    return np.random.rand(segmentation_output_size,
                          segmentation_output_size,
                          len(annotation_classes))

  segmentation_tensor = np.zeros(shape = (segmentation_output_size,
                                             segmentation_output_size,
                                             len(annotation_classes)),
                                             dtype = np.float32)
  if verbose >= 15:
    print(f"Coordinate list from within generate_augmented_segmentation_tile {coordinate_list}, length = {len(coordinate_list)}")
  level, x0, y0, downsample = coordinate_list

  x0 += x_shift
  y0 += y_shift

  # pad the area to be sampled for rotation
  x0_expanded = int(x0 - (downsample * patch_size * sqrt2min1) / 2)
  y0_expanded = int(y0 - (downsample * patch_size * sqrt2min1) / 2)
  
  # iterate through each annotation_class, generate segmentation map image,
  # and assign it to the requisite layer
  if verbose >= 100:
    print(f"{len(annotation_list)} annotations found")
  for i in range(len(annotation_classes)):
    # generate a new Image object and an Imagedraw object for the layer
    layer_image = PIL.Image.new("RGBA",  (int(patch_size * sqrt2), int(patch_size * sqrt2)), "#00000000")
    # iterate through geometries
    for j in annotation_list:
      # only draw the current geometry's name matches the current layer
      if j['properties']['classification']['name'] == annotation_classes[i]['name']:
        # only draw the polygon if it's within the bounds of the region being sampled
        if not(j['geometry']['xmax'] < x0_expanded or 
               j['geometry']['xmin'] > x0_expanded + (downsample * sqrt2 * patch_size)) and \
           not(j['geometry']['ymax'] < y0_expanded or
               j['geometry']['ymin'] > y0_expanded + (downsample * sqrt2 * patch_size)):
          coords = j['geometry']['coordinates']
          
          # check the depth of the geometry coordinates
          coord_depth = get_depth(coords)
          if coord_depth == 4:
            boundarylist = coords
          if coord_depth == 3:
            boundarylist = [coords]
          if coord_depth == 2:
            boundarylist = [[coords]]
          if coord_depth < 2 or coord_depth > 4:
            sys.exit(f"Error in processing image_id {image_id}.  Annotation {annotation_list.index(j)} coordinate depth = {coord_depth}")

          # create an aggdraw object to draw complex polygons in the PIL.Image object
          d = aggdraw.Draw(layer_image)
          brush, pen = aggdraw.Brush("white"), None

          # iterate through the boundary list
          for boundary in boundarylist:
            # create a path object for each boundary
            shape = aggdraw.Path()
            # jump the stylus on our path object to the next list of edges
            for shell in boundary:
              # move the stylus to the first point
              first_x = (shell[0][0] - x0_expanded) / downsample
              first_y = (shell[0][1] - y0_expanded) / downsample
              shape.moveto(first_x, first_y)
                # draw a line to the next stylus position in the polygonal surface
              for vertex in shell[1:]:
                next_x = (vertex[0] - x0_expanded) / downsample
                next_y = (vertex[1] - y0_expanded) / downsample
                shape.lineto(next_x, next_y)
            # close our shape object, then pass it to the draw object with our brush and pen objects, and flush it to the PIL.image object
            # from some of the mistakes I made earlier, it seems to use odd-even filling if anyone cares.
            
            shape.close()
            d.path(shape, brush, pen)
            d.flush()

    # rotate the tile 
    rotated_tile = layer_image.rotate(theta, resample = interpolation, expand = True)
    
    # crop the tile
    augmented_tile = rotated_tile.crop(((rotated_tile.size[0] - patch_size) // 2,
                                        (rotated_tile.size[1] - patch_size) // 2,
                                        (rotated_tile.size[0] + patch_size) // 2,
                                        (rotated_tile.size[1] + patch_size) // 2))
    
    augmented_tile = augmented_tile.resize((segmentation_output_size, segmentation_output_size))
    if verbose >= 100:
      plt.title(annotation_classes[i])
      plt.imshow(augmented_tile, vmin = 0.0, vmax = 1.0)
      plt.show();
    segmentation_tensor[:,:,i] = image.img_to_array(augmented_tile)[:,:,0] / 255.

  #transpose the array, if transpose flag is true
  if transposition:
    segmentation_tensor = np.transpose(segmentation_tensor, (1,0,2))

  return segmentation_tensor                 

# this is just a utilty to save a copy of all the tiles identified by the 
# whitespace_remover function to make sure they're informative, and store them
# in output_directory

def reproduce_patches(source_txt, source_wss, output_directory):

    patch_list = open(source_txt, "r")
    patch_size = int(patch_list.readline()[-4:-1])
    current_line = patch_list.readline()[0:-1]
    current_line = patch_list.readline()[0:-1]
    scan = openslide.open_slide(source_wss)
    
    while(current_line != ''):
        sys.stdout.write(source_wss + " " + current_line + "          \r")
        line_split = str.split(current_line, " ")
        image = scan.read_region((int(line_split[1],int(line_split[2])),
                                 int(line_split[0]),
                                 (patch_size, patch_size)))
        image.save(os.path.join(
            output_directory,"layer " + line_split[0] + " " +
            line_split[1] + ", " + line_split[2] + ".png"))
        current_line = patch_list.readline()[0:-1]
        
# This is a function that takes a dataframe with a column name given by 'wss_column'
# (default is "filename") with wss filenames, a directory path to the whole slide scans,
# and a patchsize (default is 512), and returns a new df with the following columns:

# 1) patch_size (given by patch size)
# 2) patch_list - a list of informative patch coordinates from generate_patch_list(),
#    note: if the source .txt is not found, it can be generated using 
#    informative_patch_finder()
# 3) number of informative patches in the wss

# If not otherwise specified, the order of the patch list is shuffled (shuffle_list)

def import_patches_to_df(df, wss_path, wss_column = 'filename',
                         patch_size = 512, shuffle_list = True, verbose = True):
    numrows, numcols = df.shape
    count = 0
    patch_size_list = []
    patch_list_list = []
    num_patches_list = []
    for wss_file in df[wss_column]:

        patch_list = informative_patch_finder(wss_file, wss_path, verbose = verbose)
        if shuffle_list:
            random.shuffle(patch_list)
        patch_list_list.append(patch_list)
        # patch_size_list.append(patch_size)
        num_patches_list.append(len(patch_list))
        patch_size_list.append(patch_size)

        if verbose:
            sys.stdout.write(f"Processing row {count + 1} of {numrows}: {wss_file} found {num_patches_list[-1]} patches\r")
        count +=1
    
    if verbose:
      print()
      print()
      if verbose > 1:
        print("Beginning import of patch elements to dataframe")
        time.sleep(verbose)
    
    df.loc[:,'patch_size'] = np.array(patch_size_list, dtype = np.uint16, copy = False)
    df.loc[:,'num_patches'] = np.array(num_patches_list, dtype = np.uint16, copy = False)
    df.loc[:,'patch_list'] = np.array(patch_list_list, dtype = object, copy = False)

# generate a list of classes from a json file.  Remove "unclassified" if present
def generate_class_list(class_json_file, verbose = False):
  with open(class_json_file) as f: 
    class_list = json.load(f)['pathClasses']
    for element in class_list:
      if element.get('name') == 'Unclassified':
        class_list.remove(element)
    if verbose:
      for element in class_list:
        print(element)
    return class_list

# Takes a processed annotation file in .json format and appends it to the annotations for the given .svs file
# Also, appends embeds the string from embed_sourcename (as a means for tracking who annotations were done by).
# This function also embeds the max and min coordinates bounding the features, to accelerate redrawing annotations.
#
# NOTE: annotations files from qpdata must be derived from qpath using a .groovy script into .json format, executed from qupath
def import_and_process_feature_list(annotation_file, embed_sourcename = None, verbose = False):
  with open(annotation_file) as f: 
    feature_list = json.load(f)
  for feature in feature_list:
    if verbose > 1:
      sys.stdout.write(f"From {annotation_file} loading {feature['properties']['classification']['name']}: ")
    flattened_coordinates = list(deepflatten(feature['geometry']['coordinates']))
    reshaped_geometry = np.reshape(flattened_coordinates,(len(flattened_coordinates) // 2, 2))
    feature['geometry']['xmin'] = int(np.min(reshaped_geometry[:,0]))
    feature['geometry']['xmax'] = int(np.max(reshaped_geometry[:,0]))
    feature['geometry']['ymin'] = int(np.min(reshaped_geometry[:,1]))
    feature['geometry']['ymax'] = int(np.max(reshaped_geometry[:,1]))
    if embed_sourcename:
      feature['properties']['source'] = str(embed_sourcename)
    if verbose > 1:
      sys.stdout.write(f"xmin {feature['geometry']['xmin']}\txmax {feature['geometry']['xmax']}\tymin {feature['geometry']['ymin']}\tymax {feature['geometry']['ymax']}\r")
  return feature_list

# if a preexisting, processed annotation file exist, this loads it and returns the list
def import_feature_list(annotation_file, verbose = False):
  if verbose:
    sys.stdout.write(f"Loading {annotation_file}")
  with open(annotation_file) as f: 
    feature_list = json.load(f)
    if verbose:
      stdout.write(f"\tComplete!\r")
    return feature_list

# Takes a list of directories with '.annotation' files and merges the annotations into single files.
# Outputs the newly generated files to the target directory with the suffix '.annot' (in json format).
# Embeds the penultimate directory folder name into the .annot file to track annotator
#
# Note: Overwrites any existing file.  
def merge_annotation_directories(inpaths, outpath, verbose = False):
  
  # listify inpaths if not already a list
  if type(inpaths) != list:
    if verbose >= 5:
      print("converting inpaths to list form")
    inpaths = [inpaths]

  # generate outpath directoy if it doesn't exist
  if not(os.path.exists(outpath)):
    if verbose:
      print(f"Generating directory: {outpath}")
    try:
      os.makedirs(outpath)
    except:
      sys.exit(f"Unable to make directory {outpath}")
  
  # files should only be created once
  created_file_list = []
  
  # iterate through the source directories
  for directory in inpaths:
    # iterate through .annotation files in source directory
    for filename in os.listdir(directory):
      if verbose >= 5:
        print(f"Processing {filename}")
      # only process a file if it of the right type, and if the file size is greater than 2 bytes (note empty),
      # and it wasn't included in a previous round of processing
      output_filename = str.split(os.path.split(filename)[1], '.')[0] + '.annot'      
      if filename.endswith(".annotations") and \
        os.path.getsize(os.path.join(directory,filename)) > 2 and \
        not(output_filename in created_file_list):
        # generate output file path
        output_filepath = os.path.join(outpath, output_filename)
        current_file_data = []
        if verbose >= 5:
          print(f"Generating {output_filepath}")
        # iterate through all directories looking for all annotation file for the same .svs
        for embed_dir_name in inpaths:
          # if a copy of the filename exists, append it to the current_file_data list
          if filename in os.listdir(embed_dir_name):
            if verbose >= 10:
              print(f"{filename} found in {embed_dir_name}")
            # generate a string for the embedding (this is to track who annotated the file)
            embed_sourcename = embed_dir_name.split('/')
            for embed_substrings in embed_sourcename:
              if embed_substrings == '':
                embed_sourcename.remove('')
            if verbose:
              sys.stdout.write(f'extending {embed_sourcename[-1]} annotations to {filename}          \r')
              filedata = import_and_process_feature_list(os.path.join(embed_dir_name,filename), 
                                                                  embed_sourcename=embed_sourcename[-1])
              current_file_data.extend(filedata)
            if verbose >= 10:
              print(f"{len(filedata)} features found in {embed_sourcename[-1]}.  Total length = {len(current_file_data)}")
        file_object = open(output_filepath, "w")
        file_object.write(json.dumps(current_file_data, indent=1))
        created_file_list.append(output_filename)

  return

# Imports all segmentation files (with the suffix '.annot') to dataframe
#   df - dataframe into which the values will be imported
#   path - directory name where .annot files are stored
#   target_colname - the name of a new column, into which annotations will be imported
#   reference_colname - the name of the column from which the svs filename will be searched
def import_segmentation_directory_to_df(df, path, target_colname, reference_colname, verbose = True):
  
  if verbose:
    print(f"Importing segmentation from {path}")
  if not(os.path.isdir(path)):
    sys.exit(f"directory {path} not found")
  if not reference_colname in df.columns:
    sys.exit(f"{reference_colname} not found in dataframe")
  
  # obtain a list of all files in the path directory

  file_list = os.listdir(path)
  if verbose >= 5:
    print(file_list)
    print()

  # initialize the target column to "None"
  segmentation_list = []
  count_cases = 0

  # iterate through the dataframe.  If a .annot file exists, import to df, otherwise set value to None
  for index, data in df.iterrows():
    found_flag = 0
    # for each element, search the file list for that element.  Also, if found, it will terminate the loop early (so duplicates won't work)
    for filename in file_list:
      # if an annotation file is found, import the annotations to the appropriate entry in the data frame
      if str(data[reference_colname]) in filename:
        with open(os.path.join(path, filename)) as f: 
          segmentation_data = list(json.load(f))
          segmentation_list.append(segmentation_data)
        found_flag = 1
        if verbose:
          sys.stdout.write(f'Imported {len(segmentation_data)} annotation objects from {filename}       \r')
          count_cases += 1
        break
    if not(found_flag):
      segmentation_list.append(None)
    
  if verbose:
    print(f"Segmentation data imported for {count_cases} cases              ")
    print()

  segmentation_array = np.array(segmentation_list, dtype = object, copy = False)

  df.loc[:,target_colname] = segmentation_array

# Return the least common multiple of a list of integers.
# Needed to automatically weight classes
def lcm(a):
  lcm = a[0]
  for i in a:
    lcm = (i * lcm)//math.gcd(i,lcm)
  return lcm

  #### This is a generator to take a dataframe ('df') and generate input and output tensors as follows:
#
# x_col      
#   a string of a column name in df containing source whole slide scans 
#   (default 'filename')
#
# y_col      
#   a string or list of strings corresponding to columns of the df.
#   Note:  For autoencoders (col_mode = 'input'), None is an acceptable column
#
# col_mode   
#   a string or list of strings that corresponds to the expected output mode
#   corresponding to each column in y_column
#   to determine the mode of encoding ('binary', 'categorical', 
#   'input', 'raw', 'segmentation', 'downsample')
#
# patch_size  
#   The size of the images generated 
#   Defaults to 512.  Tiles assumed to be square, with channels = 3 (RGB)
#
# batch_size  
#   Number of tiles and annotations to yield per step
#
# weighted_key 
#   a column to identify underrepresented elements and pad them 
#   to prevent the most represented feature from dominating the model 
#
# autoencoder_target_resize
#   If there's an "input" y_col, the size to resize the tiles to.  
#   Defaults to patch_size
#
# segmentation_class_file 
#   The path to a file in json format of classes in the annotation files
#
# segmentation_output_size
#   The size (in pixels) to rescale the augmented segmentation images
#   Defaults to autoencorder_target_resize (if specified) or patch_size 
#
# shuffle_rows
#   unless otherwise specified (set to False), shuffle rows of the dataframe
#
# overwrite
#   flag to determine whether the whitespace removed textfiles 
#   should be regenerated (if nonexistant).
#   WARNING:  This takes a VERY long time.  Defaults to False.
# 
# verbose
#   degree of verbosity.  Defaults to 1.

def wss_sample_generator(df, output_mode_log, x_col, y_col,
                         col_mode = None, 
                         patch_size = 512, 
                         batch_size = 4, 
                         weighted_key = None,
                         autoencoder_target_resize = None,
                         segmentation_classes = None, 
                         segmentation_output_size = None, 
                         missing_mode = 'random',
                         shuffle_rows = True,
                         overwrite = False,
                         verbose = False):
  

  if verbose:
        print()
        print("Initializing generator")
  if col_mode == None:
      sys.exit("Error: col_mode not specified")
  
  # save column names for checking 
  colnames = list(df.columns)
  numrows = df.shape[0]
    
  # remove rows with no patches - this usually implies no such .svs file is present
  empty_cases = 0
  empty_case_list = []
  for row, data in df.iterrows():
      if data['num_patches'] == 0:
        if verbose:
            print(f"Removing entry for {data[x_col]}.  No informative areas found")
        df = df.drop(row)
  
  # set missing values to not a number
  df.replace('', np.nan, inplace=True)

  # x is the input tensor to be fed to a neural network
  x = np.zeros((batch_size, 
                  patch_size, 
                  patch_size, 3))
  
  # y is the (list of) output tensors.  These will be processed by the types specified in
  # "col_mode."  If any of the columns for a given row is not specified, it will be assigned
  # a default value.  This should be a viable solution for entities which do not fit into a 
  # well-defined diagnostic class.  The following code serves to initialize the y outputs.
  
  # y_map is a log that allows retrieval of elements at a later point in time
  y_map = {}
  y = []
  
  # y may be a single column, or a list of columns.  If a list, len(y_col) must be 
  # the same as len(x_col)
  if type(y_col) != list:
    y_col = list([y_col])
  if type(col_mode) != list:
    col_mode = list([col_mode])
    
  # iterate through y_cols to generate a list of encoded arrays.
  # potential modes:
  # 'binary', 'categorical', 'input', 'raw', 'segmentation', 'downsample'
  for element in range(len(col_mode)):
      
    # if the mode is binary, figure out what our classes are, append a scalar to the list
    if col_mode[element] == 'binary':
      if not(y_col[element] in colnames):
        sys.exit(f"Error: {y_col[element]} not in df.columns")
      class_list = df.loc[:,y_col[element]]
      unique_classes = []
      for item in class_list:
        if not(pd.isna(item)) and not(item in unique_classes):
          unique_classes.append(item)
      if len(unique_classes) > 2:
        sys.exit(f"Error: Binary argument {y_col[element]} expects 2 targets, got {len(unique_classes)}: {unique_classes}")
      unique_classes.sort()
      bool_dict = dict(zip(unique_classes, [False, True]))
      y.append(np.zeros(batch_size, dtype=np.float32))
      if verbose >= 5:
        print(f"Created binary tensor {bool_dict}")
        # time.sleep(verbose)
      y_map.update({y_col[element] : bool_dict})
      
    # if the mode is categorical, make a list of unique categories, 
    # alphabetize, and append an array of size == number of unique categories to y
    if col_mode[element] == 'categorical':
      if not(y_col[element] in colnames):
        sys.exit(f"Error: {y_col[element]} not in df.columns")
      class_list = df.loc[:,y_col[element]]
      unique_classes = []
      for item in class_list:
        if not(pd.isna(item)) and not(item in unique_classes):
          unique_classes.append(item)
      unique_classes.sort()
      class_dict = dict(zip(unique_classes, range(len(unique_classes))))
      y.append(np.zeros((batch_size, len(unique_classes)), dtype=np.float32))
      if verbose >= 5:
        print(f"Created categorical tensor {class_dict}")
        # time.sleep(verbose)
      y_map.update({y_col[element] : class_dict})
    
    # input image, for generator, for use with autoencoders

    if col_mode[element] == 'input':
      if autoencoder_target_resize == None:
        autoencoder_target_resize = patch_size
      y_map.update({"input" : f"input image at size: {autoencoder_target_resize}, {autoencoder_target_resize}, 3"})
      y.append(np.zeros((batch_size, 
                          autoencoder_target_resize,
                          autoencoder_target_resize, 
                          3), 
                        dtype=np.float32))
      if verbose >= 5:
        print(f"Created input (autoencoder) tensor of shape ({batch_size}, {autoencoder_target_resize}, {autoencoder_target_resize}, 3)")
      
    # raw values (floating point assumed)
    if col_mode[element] == 'raw':
      if not(y_col[element] in colnames):
          sys.exit(f"Error: {y_col[element]} not in df.columns")
      y_map.update({y_col[element] : "raw_value"})
      y.append(np.zeros(batch_size))     
      if verbose >= 5:
        print(f"Created raw value tensor for {y_col['element']}")
      
    # It's unrealistic to expect every svs file to be segmented.
    # For those that are, to save of time reading and writing files and drawing polygons,
    # We will load all polygon maps, and then for each feature, we will store the
    # xmin, xmax, ymin, and ymax values.  This allows us to only draw polys which fall within tile box
    if col_mode[element] == 'segmentation':
      if segmentation_classes == None:
        sys.exit("segmentation was selected without a valid segmentation class dictionary file")
      if not(y_col[element] in colnames):
        sys.exit(f"Error: {y_col[element]} not in df.columns")
      # if segmentation_output size is not specified, set it to either autoencoder_target_resize (if specified), or patch_size
      if segmentation_output_size == None:
        if autoencoder_target_resize == None:
          segmentation_output_size = patch_size
        else:
          segmentation_output_size = autoencoder_target_resize
      # the dimensions of the segmentation vector: 
      # batchsize x patch_size x patch_size x number of features
      if verbose >= 5:
        print(f"Generating segmentation tensor of shape ({batch_size}, {segmentation_output_size}, {segmentation_output_size}, {len(segmentation_classes)})")
      y.append(np.zeros((batch_size, 
                          segmentation_output_size, 
                          segmentation_output_size, 
                          len(segmentation_classes)), 
                        dtype=np.float32))
      y_map.update({"segmentation" : segmentation_classes})

    # outputs the downsampling ratio of the field from the .svs file as a scalar float               
    if col_mode[element] == 'downsample':
      y.append(np.zeros((batch_size), dtype=np.float32))
      if verbose >= 5:
        print(f"downsample initialized to shape {np.shape(y[element])}")
      y_map.update({"downsample" : f"factor (floating point)"})

    if not(col_mode[element] in ['binary', 'categorical', 'input', 
                                  'raw', 'segmentation', 'downsample']):
      sys.exit(f"Unknown column mode: {col_mode[element]}")

  # Should implement a log system to save the classes identified within the log system,
  # and write the to a text file
  if verbose >= 5:
    print(y_map)
  logfile = open(output_mode_log, mode = 'w')
  logfile.write(json.dumps(y_map, indent=1))
  logfile.close()

  if verbose >= 15:
    print("Initializing weight section")
  
  # if weighted_class is specified, log number of occurrances of occurrances for each element
  # in that column and count them to a dictionary
  if weighted_key:
    if not(weighted_key in colnames):
      sys.exit(f"Error: Weighted class key {weighted_key} not present in dataframe")

    # an empty dict to hold our count of elements
    count_dict = {}
    
    # the list of elements from the dataframe
    key_list = df.loc[:,weighted_key]
    
    # if element is not in the dictionary, add it to the dictionary
    # for each element, increment the appropriate dictionary entry by 1
    for row in key_list:
      if not(row in count_dict):
        count_dict.update({row : 0})
      count_dict.update({row : count_dict.get(row) + 1})
    
    if verbose >= 15:
      print(f"Classes in count dictionary: {count_dict}")
    
    # compute the lcm of all elements in the dictionary
    keys_lcm = lcm(list(count_dict.values()))
    if verbose >= 15:
      print(f"LCM of {count_dict}: {keys_lcm}")
    
    # if the weighted path exists, and the overwrite flag is false,
    frequency_list = []
    augmentation_index = 0

    for index, data in df.iterrows():
      frequency_list.append(augmentation_index)
      augmentation_index += keys_lcm // count_dict.get(data[weighted_key])
    df.loc[:,'weighted_index'] = frequency_list
  else:
    if verbose >= 15:
      print("Initializing without weighted key")
    frequency_list = []
    augmentation_index = 0
    for index, data in df.iterrows():
      frequency_list.append(augmentation_index)
      augmentation_index += 1
    if verbose >= 15:
      print(f"frequency list of length {len(frequency_list)} generated")

    df.loc[:,'weighted_index'] = frequency_list

  patch_index = 0
      
  # the main yield loop, once the inputs and outputs are initialized:
  if verbose >= 5:
    print("Starting generator loop")
  while True:
    for i in range(batch_size):

      current_weighted_index = random.randint(0,augmentation_index)
      if verbose >= 15:
        print(f"Current weighted index = {current_weighted_index}")
      for index, data in df.iterrows():
        current_row = index
        if data['weighted_index'] >= current_weighted_index:
          if verbose >= 15:
            print(f"Extracting image from weighted index = {current_weighted_index}")
          break
      
      current_file = df.loc[current_row, x_col]
      if verbose >= 5:
        print(f"Generating augmented tile for {current_file}")
    
      if verbose > 1:
          sys.stdout.write(f"augmenting from {current_file}   \r")
    # generate random variables to pass to generate_augmented_tile
    # parameters: (input_wss : openslide, coordinate_list, patch_size,
    #                   x_shift = None, y_shift = None,
    #                  r_factor = None, g_factor = None, b_factor = None,
    #                    theta = None, transposition = None,
    #                    interpolation = 2):      
      j = random.randint(0,df.loc[current_row, 'num_patches'] - 1)
      coordinate_list = df.loc[current_row, "patch_list"][j]

      level = coordinate_list[0]
      downsample = coordinate_list[3]
      x_shift = random.randint(-(patch_size * downsample) // 2, 
                            (patch_size * downsample // 2))
      y_shift = random.randint(-(patch_size * downsample) // 2, 
                            (patch_size * downsample // 2))
      r_factor = random.uniform(0.9, 1.1)
      g_factor = random.uniform(0.9, 1.1)
      b_factor = random.uniform(0.9, 1.1)
      theta = random.random() * 360.0
      transposition = random.getrandbits(1)
      
      # retrieve the augmented image, rescale pixels to a range between 0.0 and 1.0
      x[i] = generate_augmented_tile(current_file, coordinate_list, patch_size, 
          x_shift, y_shift, r_factor, g_factor, b_factor, theta, transposition)
      
      if verbose >= 100:
        print(f"Generated augmented tile for {current_file}, {coordinate_list}, downsample = {downsample:.2f}, xshift = {x_shift}, y_shift = {y_shift}, r_factor = {r_factor:.2f}, g_factor = {g_factor:.2f}, b_factor = {b_factor:.2f}, theta = {theta:.2f}, transposition = {transposition}")
        augmented_tile = image.array_to_img(x[i])
        plt.imshow(augmented_tile);

      for column in range(len(col_mode)):
          
          # for input, simply assign the input image (resized if indicated) as the output image,
          # this is useful for building autoencoders to regularize our values
          if col_mode[column] == 'input':
            if verbose >= 5:
              print("generating input tile as output tile")
            if autoencoder_target_resize == patch_size:
              y[column][i] = x[i]
            else:
              y[column][i] = tf.image.resize(x[i], 
                                size = [autoencoder_target_resize, 
                                autoencoder_target_resize])
            if verbose >= 100:
              autoencoded_tile = image.array_to_img(y[column][i])
              plt.imshow(autoencoded_tile);
              
          # generates a segmented tile based on outputs
          # note that empty values are handled by the generate_augmented_segmentation_tile function
          if col_mode[column] == 'segmentation':
            if verbose >= 5:
              print(f"generating segmentated tensor of {len(segmentation_classes)} features from column {y_col[column]} with coordinates {coordinate_list}")
            y[column][i] = generate_augmented_segmentation_tile(annotation_classes=segmentation_classes,
                coordinate_list=coordinate_list,
                annotation_list=data[y_col[column]],
                missing_mode = "random",
                patch_size=patch_size,
                segmentation_output_size=segmentation_output_size,
                x_shift=x_shift,
                y_shift=y_shift,
                theta=theta,
                transposition=transposition,
                image_id = data[x_col],
                verbose=verbose)
            if verbose >= 100:
              for class_layer in range(np.shape(y[column][i])[2]):
                plt.title(segmentation_classes[class_layer])
                plt.imshow(y[column][i][:,:,class_layer], vmin = 0.0, vmax = 1.0)
                plt.show();
                   
          # one-hot encode categorical values, filling values per missing_mode
          if col_mode[column] == 'categorical':
            if verbose >= 5:
              print("Generating categorical tensor")
            if (df.loc[current_row,y_col[column]] != df.loc[current_row,y_col[column]]):
              if missing_mode == 'random':
                y[column][i,:] = np.linag(np.random.rand(shape(y[column][i])))
              if missing_mode == 'average':
                y[column][i,:] = 1.0 / len(y_map[y_col[column]])
              if missing_mode == 'ones':
                y[column][i,:] = np.ones(shape(y[column][i]))
              if missing_mode == 'zeros':
                y[column][i,:] = np.zeros(shape(y[column][i]))
            else:
              y[column][i] = keras.utils.to_categorical(
                  y_map[y_col[column]].get(df.loc[current_row,y_col[column]]),
                  len(y_map[y_col[column]]))
            if verbose >= 5:
              print(y[column][i])
                                          
          # assign missing values a value of 0.5
          # this mode will behave strangely with missing values, which needs to be addressed
          if col_mode[column] == 'binary':
            if verbose >= 5:
              print("generating binary tensor")
            if pd.isnull(df.loc[current_row,y_col[column]]) or pd.isna(df.loc[current_row,y_col[column]]) or df.loc[current_row,y_col[column]] != df.loc[current_row,y_col[column]] or df.loc[current_row,y_col[column]] == None or df.loc[current_row,y_col[column]] == "":
              if missing_mode == 'random':
                y[column][i] = np.random.rand()
              if missing_mode == 'average':
                y[column][i] == 0.5
              if missing_mode == 'ones':
                y[column][i] = 1.0
              if missing_mode == 'zeros':
                y[column][i] = 0.0
            else:
              y[column][i] = float(y_map[y_col[column]].get(df.loc[current_row,y_col[column]]))
            if verbose >= 5:
              print(y[column][i])  

          # assign missing values a value of 0.0
          if col_mode[column] == 'raw':
              if np.isnan(df.loc[current_row,y_col[column]]):
                if missing_mode == 'random':
                  y[column][i] = np.random.rand()
                if missing_mode == 'average':
                  y[column][i] = 0.5
                if missing_mode == 'ones':
                  y[column][i] = 1.0
                if missing_mode == 'zeros':
                    y[column][i] = 0.0

              else:
                  y[column][i] = float(df.loc[current_row,y_col[column]])

          if col_mode[column] == 'downsample':
              y[column][i] = coordinate_list[3]
      
      
      # if we've reached the end of the list, reset rows to 0, reshuffle list, and
      # add 1 to j (patch_list index)

    yield (x,y)