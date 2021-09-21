#importing libraries required

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

#Color identification

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image (image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    #plt.show()
    return image

def get_colors(image, number_of_colors=255, show_chart=True):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (10, 10))
        plt.title("Pi_plot of 16 colors")
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.show()
    
    return rgb_colors

get_colors(get_image('inputimage.jpg'), 16, True)

#Searching the image using color

IMG_DIR = 'images'

# We can add any number of colors required and get the specific outputs.

COLORS = {'RED': [255,0,0],'GREEN': [0,128,0],'BLUE': [0,191,255],}
images = []

for file in os.listdir(IMG_DIR):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(IMG_DIR, file)))
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])

    

def match_image_by_color(image, color, threshold = 60, number_of_colors = 15): 
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image



def show_selected_images(images, color, threshold=60, colors_to_match =15):
    index = 1
    
    for i in range(len(images)):
        selected = match_image_by_color(images[i],
                                        color,
                                        threshold,
                                        colors_to_match)
        if (selected):
            plt.subplot(1, 5, index)
            plt.imshow(images[i])
            index += 1

#Display the image having the specified color
plt.figure(figsize = (20, 10))
show_selected_images(images, COLORS['GREEN'], 60, 5)
plt.show()

