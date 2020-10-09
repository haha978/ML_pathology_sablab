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

parser = argparse.ArgumentParser()


parser.add_argument('--path', type=str, default=str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/", help='Input path')
parser.add_argument('--magnification',type=float, default=20)
parser.add_argument('--tile_size',type=int, default=224)
parser.add_argument('--output_path',type=str,default=str(pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute())+"/Desktop/")
parser.add_argument('--random_patch_number',type=int,default=100)

args = parser.parse_args()

with open(args.output_path+'names.csv', 'w') as csvfile:
    fieldnames = ['Path', 'Magnification','Tile','Tile Size','Action','No Action']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

lib = { 'Action':[], 'No Action':[] }

## Collecting the list of all svs files in a directory
#tile_name_list =  os.listdir(args.path)
tile_name_list =[]
for root, dirs, files in os.walk(args.path):
    for file in files:
        if file.endswith('.svs'):
            tile_name_list.append(root+"/"+file)

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

#all_tiles contain (tile_name,image)
def get_all_tiles(list):
    all_tiles = []
    for tile_name in list:
        #im = pyglet.resource.image("tile_"+tile_name[-10:-4]+".png")
        slide = openslide.open_slide(tile_name)
        maximum_magnification = slide.properties['openslide.objective-power']
        generator = DeepZoomGenerator(slide, tile_size=args.tile_size, overlap=0, limit_bounds=True)
        level = generator.level_count-1-int(int(slide.properties['openslide.objective-power'])/(2*args.magnification))
        [max_tile_address_x,max_tile_address_y] = np.array(generator.level_dimensions[level])/args.tile_size

        print(maximum_magnification)
        for tile in range(10):
                #this is a problem
                #try multi pooling -> order of addition really does not matter
            st_time = time.time()
                #slide = openslide.open_slide(filepath)
            bool_tile = False
            i=0
            #checks whether the tile is over certain threshold
            while bool_tile != True:
                x = np.random.choice(np.arange(int(max_tile_address_x)), 1)
                y = np.random.choice(np.arange(int(max_tile_address_y)), 1)
                temp_image = generator.get_tile(level, (x[i], y[i]))
                bool_tile = keep_tile(np.asarray(temp_image),args.tile_size,0.75)
            raw_image = temp_image.tobytes()  # tostring is deprecated
            image = pyglet.image.ImageData(temp_image.width, temp_image.height, 'RGB', raw_image)
            im = image.get_texture() #pyglet.resource.image("tile_"+tile_name[-10:-4]+".png")
            end_time = time.time()
            print('Time took to bring 1 image',(end_time-st_time))
            center_image(im)
            all_tiles.append((tile_name,im,[x[i],y[i]]))

    return all_tiles

all_tiles = get_all_tiles(tile_name_list)
print(len(all_tiles))
all_tiles_len = len(all_tiles)

"""
Define class of tile objects to keep track of tile-specific data
"""
class tile_obj:
    def __init__(self,tile_name,tile,patch_pixel):
        self.tile_name = tile_name
        self.tile = tile
        self.patch_pixel = patch_pixel
    def set_sprite(self,sprite):
        self.sprite = sprite

#initialize first two tiles that will be displayed
#as we press button NEXT
tile_obj1 = tile_obj(all_tiles[0][0],all_tiles[0][1],all_tiles[0][2])
tile_obj2 = tile_obj(all_tiles[1][0],all_tiles[1][1],all_tiles[1][2])
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
            x=game_window.width//2, y=game_window.height-300,
            anchor_x='center', batch = main_batch, group=background)

"""
draw_helper(img1,img2) takes two image files img1 and img2 and returns Sprite object
im_1 and im_2 and alligns them to center
"""
def draw_helper(img1,img2,batch):
    im_1 = pyglet.sprite.Sprite(img=img1, x=413, y=300, batch = batch, group=background)
    im_2 = pyglet.sprite.Sprite(img=img2, x=826, y=300, batch = batch, group=background)
    im_1.scale_x, im_1.scale_y = 224/im_1.width, 224/im_1.height
    im_2.scale_x, im_2.scale_y = 224/im_2.width, 224/im_2.height
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

button1 = shapes.Circle(x=413, y=150, radius = 15, batch = main_batch, group = background)
button1_checked = shapes.Circle(x=413, y=150, radius = 12, batch = main_batch,
        color=[255,255,255], group = foreground)

button2 = shapes.Circle(x=826, y=150, radius = 15, batch = main_batch, group = background)
button2_checked = shapes.Circle(x=826, y=150, radius = 12,
        batch = main_batch ,color=[255,255,255], group = foreground)


@game_window.event
def on_draw():
    global tile_obj1,tile_obj2
    global class_text,class_label, image_batch
    game_window.clear()
    main_batch.draw()
    image_batch.draw()
    # new_batch = pyglet.graphics.Batch()
    # sprite_i,sprite_j = draw_helper(tile_i,tile_j,new_batch)
    # new_batch.draw()

def in_circle(x,y,x_center,y_center,radius):
    return (x-x_center)**2+(y-y_center)**2 <= radius**2

def in_box(x,y,x_center,y_center,x_width,y_height):
    left_x, right_x = (x_center - x_width/2), (x_center + x_width/2)
    bottom_y, top_y = (y_center - y_height/2), (y_center + y_height/2)
    return left_x < x and x < right_x and bottom_y < y and y < top_y

def save_entry (path,tile,action,no_action):
    with open(args.output_path + 'names.csv', 'a', newline='') as csvfile:
        fieldnames = ['Path', 'Magnification', 'Tile', 'Tile Size', 'Action', 'No Action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow({'Path': path, "Magnification": args.magnification,
                         "Tile": tile, 'Tile Size': args.tile_size, "Action": action, "No Action": no_action})
@game_window.event
def on_mouse_press(x, y, button, modifiers):
    global tile_obj1, tile_obj2
    global class_text,class_label, image_batch, button2_checked, button1_checked
    if in_circle(x,y,826,150,15):
        if button == mouse.LEFT:
            if button2_checked.color == [255,255,255]:
                button2_checked.color = [0,0,0]
            elif button2_checked.color == [0,0,0]:
                button2_checked.color = (255,255,255)
    elif in_circle(x,y,413,150,15):
        if button == mouse.LEFT:
            if button1_checked.color == [255,255,255]:
                button1_checked.color = [0,0,0]
            elif button1_checked.color == [0,0,0]:
                button1_checked.color = [255,255,255]
    elif in_box(x,y,game_window.width//2,30,50,30):
        print('I am clicking next block')
        #save the result

        if class_text == "Please select the tiles with action.":
            if button1_checked.color == [0,0,0]:
                lib['Action'].append(tile_obj1.tile_name)
                save_entry(tile_obj1.tile_name, tile_obj1.patch_pixel,"true","false")
            else:
                lib['No Action'].append(tile_obj2.tile_name)
                save_entry(tile_obj2.tile_name, tile_obj2.patch_pixel, "false", "true")
            if button2_checked.color == [0,0,0]:
                lib['Action'].append(tile_obj1.tile_name)
                save_entry(tile_obj1.tile_name, tile_obj1.patch_pixel, "true", "false")
            else:
                lib['No Action'].append(tile_obj2.tile_name)
                save_entry(tile_obj1.tile_name, tile_obj2.patch_pixel, "false", "true")


        #reset buttons
        button1_checked.color = [255,255,255]
        button2_checked.color = [255,255,255]
        #reset pictures
        delete_sprites(tile_obj1.sprite,tile_obj2.sprite,all_tiles)
        print(lib)
        #set a new_text box
        class_label.delete()
        class_text = "Please select the tiles with action."
        class_label = pyglet.text.Label(text = class_text,
                    x=game_window.width//2, y=game_window.height-300,
                    anchor_x='center', batch = main_batch, group=foreground)
        tile_obj1.tile_name = all_tiles[0][0]
        tile_obj2.tile_name = all_tiles[1][0]
        tile_obj1.tile = all_tiles[0][1]
        tile_obj2.tile = all_tiles[1][1]
        tile_obj1.patch_pixel = all_tiles[0][2]
        tile_obj2.patch_pixel = all_tiles[1][2]
        #drawthem
        all_tiles_len = len(all_tiles)
        sprite_i,sprite_j = draw_helper(tile_obj1.tile,tile_obj2.tile,image_batch)
        tile_obj1.sprite = sprite_i
        tile_obj2.sprite = sprite_j


if __name__ == '__main__':
    pyglet.gl.glClearColor(0.5,0.7,0.7,1)
    pyglet.app.run()


if __name__ == '__main__':
    pyglet.gl.glClearColor(0.5,0.7,0.7,1)
    pyglet.app.run()
