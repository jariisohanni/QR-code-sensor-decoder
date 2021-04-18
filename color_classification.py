from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import os
import skimage

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def classify(image,number_of_colors):

    #print("The type of this input is {}".format(type(image)))
    #print("Shape: {}".format(image.shape))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   ## plt.imshow(image)

    n_colors = 4

    arr = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    modified_image = centers[labels].reshape(image.shape).astype('uint8')

    #modified_image = cv2.blur(image, (10, 10)) 

    plt.imshow(modified_image)

    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray_image, cmap='gray')

    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = n_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)


    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    #color2_rgb = sRGBColor(ordered_colors[1][0]/255.0, ordered_colors[1][1]/255.0, ordered_colors[1][2]/255.0)
    #color2_lab = convert_color(color2_rgb , LabColor)

    #color3_rgb = sRGBColor(ordered_colors[2][0]/255.0, ordered_colors[2][1]/255.0, ordered_colors[2][2]/255.0)
    #color3_lab = convert_color(color3_rgb , LabColor)

    #delta = delta_e_cie2000(color2_lab, color3_lab);  

    pix_amounts = [counts[0],counts[1],counts[2],counts[3]]
    pix_amounts.sort(reverse=True)

    total = counts[0] + counts[1] + counts[2]
    avg_count = pix_amounts[2]/total

    #sensor_to_sencond = pix_amounts[0]/pix_amounts[1]
   # if sensor_to_sencond > 0.4:
    #    return True

    #plt.figure(figsize = (8, 6))
    #plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    #plt.show()

    return avg_count

    #plt.figure(figsize = (8, 6))
    #plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    #plt.show()

    # Convert from RGB to Lab Color Space
    #color1_rgb = sRGBColor(rgb_colors[0][0]/255.0, rgb_colors[0][1]/255.0, rgb_colors[0][2]/255.0)
    #color1_lab = convert_color(color1_rgb, LabColor)

    # Convert from RGB to Lab Color Space
    #color2_rgb = sRGBColor(rgb_colors[1][1]/255.0, rgb_colors[1][1]/255.0, rgb_colors[1][2]/255.0)
    #color2_lab = convert_color(color2_rgb , LabColor)

    #color3_rgb = sRGBColor(rgb_colors[2][1]/255.0, rgb_colors[2][1]/255.0, rgb_colors[2][2]/255.0)
    #color3_lab = convert_color(color3_rgb , LabColor)

    #darkest_color = color1_lab
    #if darkest_color.lab_l > color2_lab.lab_l:
    #     darkest_color = color2_lab
    #if darkest_color.lab_l > color3_lab.lab_l:
    #     darkest_color = color3_lab 
    
    #brigthess_color = color1_lab
    #if brigthess_color.lab_l < color2_lab.lab_l:
    #     brigthess_color = color2_lab
    #if brigthess_color.lab_l < color3_lab.lab_l:
    #     brigthess_color = color3_lab 

    #delta_max = delta_e_cie2000(darkest_color, brigthess_color);  

    #if color1_lab != brigthess_color and color1_lab != darkest_color:
    #    sensor_color = color1_lab

    #if color2_lab != brigthess_color and color2_lab != darkest_color:
    #    sensor_color = color2_lab

    #if color3_lab != brigthess_color and color3_lab != darkest_color:
    #    sensor_color = color3_lab

    #delta_sensor2dark = delta_e_cie2000(darkest_color, sensor_color);  
    #delta_sensor2bright = delta_e_cie2000(brigthess_color, sensor_color);  

    #return rgb_colors
