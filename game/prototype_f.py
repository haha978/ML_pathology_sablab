import pyglet
import os
from pyglet import image
from pyglet.window import mouse
import pyglet.shapes as shapes
import random
import time
import argparse
import csv
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
import numpy as np
import openslide
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import pathlib
from PIL import Image
import time

print('Please type the input path: ')
input_path = input()
print('Please type the output path: ')
output_path = input()
print('Please type the number of action tiles required for each slide: ')
NUM_ACTION_TILES = input()
NUM_ACTION_TILES=int(NUM_ACTION_TILES)
#input_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/"
#output_path =  str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/"

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default= input_path, help='Input path')
parser.add_argument('--magnification',type=float, default=2.5)
parser.add_argument('--tile_size',type=int, default=512)
parser.add_argument('--output_path',type=str,default= output_path)
parser.add_argument('--patch_number',type=int,default=2)
parser.add_argument('--max_magnification',type=int, default=10)

"""
Save all the coordinates of the tiles of WSIs for a given magnification
in a give output_path. Name the numpy dictionary as slide_coord.npy
"""
def save_coord(path,output_path,tile_size, magnification):
    slide_name_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.svs') and root+"/"+file:
                slide_name_list.append(root+"/"+file)
    #dict_list saves slide specific data such as tile coordinates
    if os.path.isfile(os.path.join(output_path, 'slide_coord.npy')):
        dict_list = list(np.load( os.path.join(output_path, 'slide_coord.npy'), allow_pickle=True))
        completed_numpy = list(np.load( os.path.join(output_path, 'slide_coord_numpy.npy'), allow_pickle=True))
        completed_numpy_names=[]
        for numpy_name in range(len(completed_numpy)):
            completed_numpy_names.append(completed_numpy[numpy_name]['slide name'])
    else:
        dict_list=[]
        completed_numpy=[]
        completed_numpy_names = []
    coordinates = []
    for slide_name in slide_name_list:
        if os.path.isfile(output_path + "/Svs_to_numpy/" + slide_name[slide_name.rfind("/") + 1:-4] + "/useful_tile_list.csv") and slide_name not in completed_numpy_names:
            coordinates = []
            with open(output_path + "/Svs_to_numpy/" + slide_name[slide_name.rfind("/") + 1:-4] + "/useful_tile_list.csv",
                      newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    col = int(row['Tile_address_x'])
                    row = int(row['Tile_address_y'])
                    coordinates.append((col, row))

            #slide = openslide.open_slide(slide_name)
            #maximum_magnification = slide.properties['openslide.objective-power']
            #generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
            #obtain magnification
            #if magnification != int(slide.properties['openslide.objective-power']):
            #    level = generator.level_count-1-int(np.sqrt(np.float(slide.properties['openslide.objective-power'])/(magnification)))
            #else:
            #    level = generator.level_count-1
            #obtain random coordinates
            #cols ,rows = generator.level_tiles[level]
            #coordinates = []
            #for col in range(cols):
            #    for row in range(rows):
            #        coordinates.append((col,row))
            random.shuffle(coordinates)
            dict = {'slide name': slide_name, 'level': None , 'tile size': tile_size, 'tile addresses': coordinates}
            dict_list.append(dict)
            completed_numpy.append(dict)
    np.save(os.path.join(output_path,'slide_coord.npy'),dict_list)
    np.save(os.path.join(output_path, 'slide_coord_numpy.npy'), completed_numpy)

args = parser.parse_args()
NEXT_STATE = 0
output_files = os.listdir(args.output_path)
#The number of action tiles we want

#create numpy dictionary of WSIs if there are no npy initialized
#if 'slide_coord_numpy.npy' not in output_files:
save_coord(args.path, args.output_path, args.tile_size, args.magnification)
print("Done creating metadata for the Whole Slide Images")

#now we have the slide_coord.npy saved
#dict array saves all the metadata for slides. Will modify this as the game progresses
DICT_ARRAY_PATH = os.path.join(args.output_path,'slide_coord.npy')
DICT_ARRAY = np.load( DICT_ARRAY_PATH, allow_pickle=True)
Temp_NUM_ACTION_TILES = 0
#Check whether there is a csvfile to save
csv_path = os.path.join(args.output_path,'Annotations.csv')
if 'Annotations.csv' not in output_files:
    with open(csv_path, 'w') as csvfile:
        fieldnames = ['Path', 'Magnification','Tile','Tile Size','Action','No Action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
else:
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

lib = { 'Action':[], 'No Action':[] }

## Collecting the list of all svs files in a directory
#tile_name_list =  os.listdir(args.path)
tile_name_list =[]
count=0
for root, dirs, files in os.walk(args.path):
    for file in files:
        if file.endswith('.svs') and root+"/"+file:
            tile_name_list.append(root+"/"+file)

#print(tile_name_list)
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

## A placeholder to store number of action patches per svs defined by the pathologist
action_patch = np.zeros(len(tile_name_list))

#set game window
game_window = pyglet.window.Window(1240, 800)

#define cursor
cursor = game_window.get_system_mouse_cursor(game_window.CURSOR_HAND)
game_window.set_mouse_cursor(cursor)

#define resource, which is where I pull the tiles
pyglet.resource.path = [args.path]
pyglet.resource.reindex()

#get center of an image
def center_image(image):
     image.anchor_x = image.width // 2
     image.anchor_y = image.height // 2

"""
This get_tiles function will be repeatedly called
during the game to add the tiles to the all_tiles
"""
def get_tiles(patch_number,tile_address):
    global DICT_ARRAY, DICT_ARRAY_PATH, Temp_NUM_ACTION_TILES
    tiles = []
    dict1 = DICT_ARRAY[0]
    #check whether there are coordinates in a slide we are interested

    #Now we are sure tha the slide we are interested have some tiles

    #NOW I have my level obtained
    if tile_address == None:
        if Temp_NUM_ACTION_TILES >= NUM_ACTION_TILES or len(dict1['tile addresses']) < 2:
            DICT_ARRAY = DICT_ARRAY[1:]
            np.save(DICT_ARRAY_PATH, DICT_ARRAY, allow_pickle=True)
            dict1 = DICT_ARRAY[0]
            Temp_NUM_ACTION_TILES = 0
        #level = dict1['level']
        # If tile_address == None, means I am at 2.5x
        # max_x is column and max_y are rows
        magnification = 2.5
        #max_x,max_y = generator.level_tiles[level]
        for tile in range(patch_number):
            #bool_tile = False
            #checks whether the tile is over certain threshold
            #while bool_tile != True:
            tile_addresses = dict1['tile addresses']
            col, row = tile_addresses.pop()
                #Need to save the modified dict1
            dict1['tile addresses'] = tile_addresses
            DICT_ARRAY[0] = dict1
            np.save(DICT_ARRAY_PATH,DICT_ARRAY,allow_pickle=True)
            temp_image = np.load(output_path + "/Svs_to_numpy/" + dict1['slide name'][dict1['slide name'].rfind("/") + 1:-4]+"/Tile_x_"+str(col)+"_Tile_y_"+str(row)+".npy" )
            #temp_image = generator.get_tile(level, (col, row))
            temp_image=Image.fromarray(np.uint8(temp_image))
            #bool_tile = keep_tile(np.asarray(temp_image),args.tile_size,0.75)
            #obtain patches in raw image
            raw_image = temp_image.tobytes()  # tostring is deprecated
            image = pyglet.image.ImageData(temp_image.width, temp_image.height, 'RGB', raw_image)
            im = image.get_texture() #pyglet.resource.image("tile_"+tile_name[-10:-4]+".png")
            end_time = time.time()
            center_image(im)
            tiles.append((dict1['slide name'],im,[col,row],magnification))
    else:
        magnification = 10
        slide = openslide.open_slide(dict1['slide name'])
        maximum_magnification = slide.properties['openslide.objective-power']
        generator = DeepZoomGenerator(slide, tile_size=args.tile_size, overlap=0)
        if magnification != int(slide.properties['openslide.objective-power']):
            level = generator.level_count-1-int(np.sqrt(np.float(slide.properties['openslide.objective-power'])/(magnification)))
        else:
            level = generator.level_count-1

        [[min_tile_address_x, max_tile_address_x],[min_tile_address_y, max_tile_address_y]]= tile_address
        x = np.arange(min_tile_address_x, max_tile_address_x, 1)
        y = np.arange(min_tile_address_y, max_tile_address_y, 1)
        print(x)
        print(y)
        for j in range(4):
            for i in range(4):
                bool_tile = False
                temp_image = generator.get_tile(level, (x[i], y[j]))
                raw_image = temp_image.tobytes()  # tostring is deprecated
                image = pyglet.image.ImageData(temp_image.width, temp_image.height, 'RGB', raw_image)
                im = image.get_texture()  # pyglet.resource.image("tile_"+tile_name[-10:-4]+".png")
                center_image(im)
                tiles.append((dict1['slide name'], im, [x[i], y[j]], magnification))
                print(len(tiles))
    return tiles

print(DICT_ARRAY)
all_tiles = get_tiles(args.patch_number,None)
#print(len(all_tiles))
all_tiles_len = len(all_tiles)

"""
Define class of tile objects to keep track of tile-specific data
"""
class tile_obj:
    def __init__(self,tile_name,tile,patch_pixel,magnification):
        self.tile_name = tile_name
        self.tile = tile
        self.patch_pixel = patch_pixel
        self.magnification = magnification
    def set_sprite(self,sprite):
        self.sprite = sprite

#initialize first two tiles that will be displayed
#as we press button NEXT
tile_obj1 = tile_obj(all_tiles[0][0],all_tiles[0][1],all_tiles[0][2],all_tiles[0][3])
tile_obj2 = tile_obj(all_tiles[1][0],all_tiles[1][1],all_tiles[1][2],all_tiles[0][3])
#create main_batch for background and text labels
main_batch = pyglet.graphics.Batch()
#initialize image_batch
image_batch = pyglet.graphics.Batch()
#incase we need to distinguish what comes up front
background = pyglet.graphics.OrderedGroup(0)
foreground = pyglet.graphics.OrderedGroup(1)
#create label for the game_window
level_label = pyglet.text.Label(text = "Action / No Action Game",
        x=game_window.width//2, y=game_window.height-30,
        anchor_x='center', batch = main_batch, group=foreground)
#create class label
class_text = "Please select the tiles with action."
class_label = pyglet.text.Label(text = class_text,
            x=game_window.width//2, y=game_window.height-80,
            anchor_x='center', batch = main_batch, group=background)

tile_information_1 = pyglet.text.Label(text = tile_obj1.tile_name[tile_obj1.tile_name.rfind("-")+1:-4]+"  "+str(tile_obj1.patch_pixel)+"  "+str(tile_obj1.magnification),
            x=313, y=100,
            anchor_x='center', batch = main_batch, group=background)

tile_information_2 = pyglet.text.Label(text = tile_obj2.tile_name[tile_obj2.tile_name.rfind("-")+1:-4]+"  "+str(tile_obj2.patch_pixel)+"  "+str(tile_obj2.magnification),
            x=926, y=100,
            anchor_x='center', batch = main_batch, group=background)

"""
draw_helper(img1,img2) takes two image files img1 and img2 and returns Sprite object
im_1 and im_2 and alligns them to center
"""
def draw_helper(img1,img2,batch):
    im_1 = pyglet.sprite.Sprite(img=img1, x=313, y=450, batch = batch, group=background)
    im_2 = pyglet.sprite.Sprite(img=img2, x=926, y=450, batch = batch, group=background)
    im_1.scale_x, im_1.scale_y = 512/im_1.width, 512/im_1.height
    im_2.scale_x, im_2.scale_y = 512/im_2.width, 512/im_2.height
    return im_1, im_2

"""
delete_sprites(img1) takes two SPRITE objects and deletes them.
"""
def delete_sprites(im_1,im_2,all_tiles):
    im_1.delete()
    im_2.delete()
    all_tiles.pop(0)
    all_tiles.pop(0)


sprite_i,sprite_j = draw_helper(tile_obj1.tile,tile_obj2.tile,image_batch)
tile_obj1.set_sprite(sprite_i)
tile_obj2.set_sprite(sprite_j)

"""
Initialize buttons and boxes available
"""
next_box = shapes.Rectangle(x=game_window.width//2, y = 30, batch = main_batch, color =(100, 100, 255), width = 50, height=30, group = background)

#quit_box = shapes.Rectangle(x=3*game_window.width//4, y = 35, batch = main_batch, color =(100, 100, 255), width = 50, height=30, group = background)

center_image(next_box)
next_text = pyglet.text.Label(text = 'NEXT',x=game_window.width//2, y = 30,
        batch = main_batch, anchor_x = 'center',anchor_y = 'center', color = (255,255,255,255) ,group = foreground)

#quit_text = pyglet.text.Label(text = 'QUIT',x=3*game_window.width//4, y = 35,
 #       batch = main_batch,  color = (255,255,255,255) ,group = foreground)

button1 = shapes.Circle(x=313, y=150, radius = 15, batch = main_batch, group = background)
button1_checked = shapes.Circle(x=313, y=150, radius = 12, batch = main_batch,
        color=[255,255,255], group = foreground)

button2 = shapes.Circle(x=926, y=150, radius = 15, batch = main_batch, group = background)
button2_checked = shapes.Circle(x=926, y=150, radius = 12,
        batch = main_batch ,color=[255,255,255], group = foreground)


@game_window.event
def on_draw():
    global tile_obj1,tile_obj2,tile_obj3,tile_obj4
    global class_label, image_batch, tile_information_1,tile_information_2
    game_window.clear()
    main_batch.draw()
    image_batch.draw()

def in_circle(x,y,x_center,y_center,radius):
    return (x-x_center)**2+(y-y_center)**2 <= radius**2

def in_box(x,y,x_center,y_center,x_width,y_height):
    left_x, right_x = (x_center - x_width/2), (x_center + x_width/2)
    bottom_y, top_y = (y_center - y_height/2), (y_center + y_height/2)
    return left_x < x and x < right_x and bottom_y < y and y < top_y

def save_entry (path,tile,action,no_action,magnification):
    with open(os.path.join(args.output_path, 'Annotations.csv' ), 'a', newline='') as csvfile:
        fieldnames = ['Path', 'Magnification', 'Tile', 'Tile Size', 'Action', 'No Action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Path': path, "Magnification": magnification,
                         "Tile": tile, 'Tile Size': args.tile_size, "Action": action, "No Action": no_action})
@game_window.event
def on_mouse_press(x, y, button, modifiers):
    global tile_obj1, tile_obj2, NEXT_STATE,Temp_NUM_ACTION_TILES
    global class_text,class_label, image_batch, button2_checked, button1_checked, tile_information_1,tile_information_2
    if in_circle(x,y,926,150,15):
        if button == mouse.LEFT:
            if button2_checked.color == [255,255,255]:
                button2_checked.color = [0,0,0]
            elif button2_checked.color == [0,0,0]:
                button2_checked.color = (255,255,255)
    elif in_circle(x,y,313,150,15):
        if button == mouse.LEFT:
            if button1_checked.color == [255,255,255]:
                button1_checked.color = [0,0,0]
            elif button1_checked.color == [0,0,0]:
                button1_checked.color = [255,255,255]
    elif in_box(x,y,game_window.width//2,30,50,30):
        if NEXT_STATE == 0:
            NEXT_STATE = 1
            if button1_checked.color == [0,0,0]:
                lib['Action'].append(tile_obj1.tile_name)
                Temp_NUM_ACTION_TILES+=1
                save_entry(tile_obj1.tile_name, tile_obj1.patch_pixel,"true","false",tile_obj1.magnification)
                patch_number=16
                tile_address=[[4*tile_obj1.patch_pixel[0],4*tile_obj1.patch_pixel[0]+4],[4*tile_obj1.patch_pixel[1],4*tile_obj1.patch_pixel[1]+4]]
                if (tile_obj1.magnification*4<=args.max_magnification):
                    t0 = time.time()
                    additional_tiles = get_tiles(patch_number,tile_address)
                    print(str(time.time()-t0)+" seconds to bring 16 10x tiles.")
                    for i in range(patch_number):
                        all_tiles.insert(2, additional_tiles[15-i])
            else:
                lib['No Action'].append(tile_obj1.tile_name)
                save_entry(tile_obj1.tile_name, tile_obj1.patch_pixel, "false", "true",tile_obj1.magnification)

            if button2_checked.color == [0,0,0]:
                lib['Action'].append(tile_obj2.tile_name)
                Temp_NUM_ACTION_TILES += 1
                save_entry(tile_obj2.tile_name, tile_obj2.patch_pixel, "true", "false",tile_obj2.magnification)
                #a = np.argwhere(np.array(all_tiles) == tile_obj2.tile_name)
                patch_number=16
                tile_address = [[4 * tile_obj2.patch_pixel[0], 4* tile_obj2.patch_pixel[0] + 4],
                                [4 * tile_obj2.patch_pixel[1], 4 * tile_obj2.patch_pixel[1] + 4]]
                if (tile_obj2.magnification * 4 <= args.max_magnification):
                    t0 = time.time()
                    additional_tiles = get_tiles(patch_number,tile_address)
                    print(str(time.time() - t0) + " seconds to bring 16 10x tiles.")
                    for i in range(patch_number):
                        all_tiles.insert(2, additional_tiles[15-i])
            else:
                lib['No Action'].append(tile_obj2.tile_name)
                save_entry(tile_obj2.tile_name, tile_obj2.patch_pixel, "false", "true",tile_obj2.magnification)

            #reset buttons
            button1_checked.color = [255,255,255]
            button2_checked.color = [255,255,255]
            #reset pictures
            delete_sprites(tile_obj1.sprite,tile_obj2.sprite,all_tiles)
            #print(lib)
            #Obtain new 2.5x tiles if all_tiles is empty
            if all_tiles == []:
                additional_tiles = get_tiles(args.patch_number, None)
                all_tiles.extend(additional_tiles)
            tile_obj1.tile_name = all_tiles[0][0]
            tile_obj2.tile_name = all_tiles[1][0]
            tile_obj1.tile = all_tiles[0][1]
            tile_obj2.tile = all_tiles[1][1]
            tile_obj1.patch_pixel = all_tiles[0][2]
            tile_obj2.patch_pixel = all_tiles[1][2]
            tile_obj1.magnification = all_tiles[0][3]
            tile_obj2.magnification = all_tiles[1][3]
            #drawthem
            all_tiles_len = len(all_tiles)
            sprite_i,sprite_j = draw_helper(tile_obj1.tile,tile_obj2.tile,image_batch)
            tile_obj1.sprite = sprite_i
            tile_obj2.sprite = sprite_j
            tile_information_1.delete()
            tile_information_1 = pyglet.text.Label(
                text=tile_obj1.tile_name[tile_obj1.tile_name.rfind("-")+1:-4] + "  " + str(
                    tile_obj1.patch_pixel) + "  " + str(tile_obj1.magnification),
                x=313, y=100,
                anchor_x='center', batch=main_batch, group=background)
            tile_information_2.delete()
            tile_information_2 = pyglet.text.Label(
                text=tile_obj2.tile_name[tile_obj2.tile_name.rfind("-")+1:-4] + "  " + str(
                    tile_obj2.patch_pixel) + "  " + str(tile_obj2.magnification),
                x=926, y=100,
                anchor_x='center', batch=main_batch, group=background)
            NEXT_STATE = 0

if __name__ == '__main__':
    pyglet.gl.glClearColor(0.5,0.7,0.7,1)
    pyglet.app.run()
