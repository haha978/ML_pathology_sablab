import pyglet
import os
from pyglet import image
from pyglet.window import mouse
import pyglet.shapes as shapes
import random
import time
import argparse
import csv
import numpy as np
import openslide
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator


parser = argparse.ArgumentParser()


parser.add_argument('--path', type=str, default='/Users/caglabahadir/Desktop/Prototype/', help='Input path')
parser.add_argument('--magnification',type=float, default=10)
parser.add_argument('--tile_size',type=int, default=256)
parser.add_argument('--output_path',type=str,default="/Users/caglabahadir/Desktop/Prototype/")

args = parser.parse_args()
#def main():

with open(args.output_path+'names.csv', 'w') as csvfile:
    fieldnames = ['Path', 'Magnification','Tile','Action','No Action']
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

## A placeholder to store number of action patches per svs defined by the pathologist
action_patch = np.zeros(len(tile_name_list))

#set game window
game_window = pyglet.window.Window(1240, 800)

#define cursor
cursor = game_window.get_system_mouse_cursor(game_window.CURSOR_HAND)
game_window.set_mouse_cursor(cursor)

#define resource, which is where I pull the tiles
st_time = time.time()
pyglet.resource.path = [args.path]
pyglet.resource.reindex()
end_time = time.time()
print('Time took to make resource path',(end_time-st_time))

#get center of an image
def center_image(image):
     image.anchor_x = image.width // 2
     image.anchor_y = image.height // 2

#all_tiles contain (tile_name,image)
def get_all_tiles(list):
    all_tiles = []
    for tile_name in list:
        #this is a problem
        #try multi pooling -> order of addition really does not matter
        st_time = time.time()
        #slide = openslide.open_slide(filepath)
        slide = openslide.open_slide(tile_name)
        generator = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
        ima = generator.get_tile(10, (0, 0))
        ima.save("/Users/caglabahadir/Desktop/Prototype/tile_"+tile_name[-10:-4]+".png")
        im = pyglet.resource.image("tile_"+tile_name[-10:-4]+".png")
        end_time = time.time()
        print('Time took to bring 1 image',(end_time-st_time))
        center_image(im)
        all_tiles.append((tile_name,im))
    return all_tiles
#want to get (tile_name, image) list of all tiles available

all_tiles = get_all_tiles(tile_name_list)

all_tiles_len = len(all_tiles)
#initialize first two tiles that will be displayed, tile_name_i tile_name_j will change
#as we press button NEXT
tile_name_i = all_tiles[0][0]
tile_name_j = all_tiles[1][0]
tile_i = all_tiles[0][1]
tile_j = all_tiles[1][1]
#create main_batch for background and text labels
main_batch = pyglet.graphics.Batch()
#initialize image_batch
image_batch = pyglet.graphics.Batch()
#incase we need to distinguish what comes up front
background = pyglet.graphics.OrderedGroup(0)
foreground = pyglet.graphics.OrderedGroup(1)
#create label for the game_window
level_label = pyglet.text.Label(text = "My Amazing Game",
        x=game_window.width//2, y=game_window.height-30,
        anchor_x='center', batch = main_batch, group=foreground)
#create class label
class_text = "Action"
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

sprite_i,sprite_j = draw_helper(tile_i,tile_j,image_batch)
"""
Initialize buttons and boxes available
"""
next_box = shapes.Rectangle(x=game_window.width//2, y = 30, batch = main_batch, color =(100, 100, 255), width = 50, height=30, group = background)

quit_box = shapes.Rectangle(x=3*game_window.width//4, y = 35, batch = main_batch, color =(100, 100, 255), width = 50, height=30, group = background)

center_image(next_box)
next_text = pyglet.text.Label(text = 'NEXT',x=game_window.width//2, y = 30,
        batch = main_batch, anchor_x = 'center',anchor_y = 'center', color = (255,255,255,255) ,group = foreground)

quit_text = pyglet.text.Label(text = 'QUIT',x=3*game_window.width//4, y = 35,
        batch = main_batch,  color = (255,255,255,255) ,group = foreground)

button1 = shapes.Circle(x=413, y=150, radius = 15, batch = main_batch, group = background)
button1_checked = shapes.Circle(x=413, y=150, radius = 12, batch = main_batch,
        color=[255,255,255], group = foreground)

button2 = shapes.Circle(x=826, y=150, radius = 15, batch = main_batch, group = background)
button2_checked = shapes.Circle(x=826, y=150, radius = 12,
        batch = main_batch ,color=[255,255,255], group = foreground)


@game_window.event
def on_draw():
    global sprite_i, sprite_j, tile_name_i, tile_name_j,tile_i,tile_j,all_tiles_len
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
        fieldnames = ['Path', 'Magnification', 'Tile', 'Action', 'No Action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow({'Path': path, "Magnification": args.magnification,
                         "Tile": tile, "Action": action, "No Action": no_action})
@game_window.event
def on_mouse_press(x, y, button, modifiers):
    global sprite_i, sprite_j, tile_name_i, tile_name_j,tile_i,tile_j,all_tiles_len
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
        path="hey"
        path_2="hey"
        tile="my"
        tile_2="my_2"
        if class_text == 'Action':
            if button1_checked.color == [0,0,0]:
                lib['Action'].append(tile_name_i)
                save_entry(path,tile,"true","false")
            else:
                lib['No Action'].append(tile_name_i)
                save_entry(path, tile, "false", "true")
            if button2_checked.color == [0,0,0]:
                lib['Action'].append(tile_name_j)
                save_entry(path_2, tile_2, "true", "false")
            else:
                lib['No Action'].append(tile_name_j)
                save_entry(path_2, tile_2, "false", "true")


        #reset buttons
        button1_checked.color = [255,255,255]
        button2_checked.color = [255,255,255]
        #reset pictures
        delete_sprites(sprite_i,sprite_j,all_tiles)
        print(lib)
        #set a new_text box
        class_label.delete()
        class_text = "Action"
        class_label = pyglet.text.Label(text = class_text,
                    x=game_window.width//2, y=game_window.height-300,
                    anchor_x='center', batch = main_batch, group=foreground)
        tile_name_i = all_tiles[0][0]
        tile_name_j = all_tiles[1][0]
        tile_i = all_tiles[0][1]
        tile_j = all_tiles[1][1]
        #drawthem
        all_tiles_len = len(all_tiles)
        sprite_i,sprite_j = draw_helper(tile_i,tile_j,image_batch)


if __name__ == '__main__':
    pyglet.gl.glClearColor(0.5,0.7,0.7,1)
    pyglet.app.run()