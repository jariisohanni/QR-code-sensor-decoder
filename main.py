
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
import qrcode
import string
import math
import color_classification
import os
import os, re

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def point_pos(x0, y0, d, theta):
    #theta_rad = math.pi/2 - math.radians(theta)
    return x0 + d*math.cos(theta), y0 + d*math.sin(theta)

def calculateSensorRect(corner_points, cells):

    x0 = corner_points[0][0][0] #tl
    y0 = corner_points[0][0][1]

    x1 = corner_points[0][1][0] #tr
    y1 = corner_points[0][1][1]

    x2 = corner_points[0][2][0] #br
    y2 = corner_points[0][2][1]

    x3 = corner_points[0][3][0] #bl
    y3 = corner_points[0][3][1]

    height = math.sqrt( ((y0-y1)**2)+((x0-x1)**2) )

    width = math.sqrt( ((x0-x3)**2)+((y0-y3)**2) )
  
    angle_x = math.atan2(y1-y0, x1-x0)
    angle_y = math.atan2(y3-y0, x3-x0)

    one_row_x = width / cells
    one_row_y = height / cells

    sh = (2*one_row_y)
    sy = (one_row_y*(cells-3))

    sw = (one_row_x * (cells-9))
    sx = (one_row_x * 8)

    sensor_bl = point_pos(x3,y3,one_row_x*8,angle_x)

    sensor_tl = point_pos(sensor_bl[0],sensor_bl[1],one_row_y*-5,angle_y)

    sensor_tr = point_pos(sensor_tl[0],sensor_tl[1],sw,angle_x)

    sensor_br = point_pos(x3,y3,one_row_x*cells,angle_x)

    r = (
        round(sensor_tl[0]),
        round(sensor_tl[1]),
        round(sensor_tr[0]),
        round(sensor_tr[1]),
        round(sensor_br[0]),
        round(sensor_br[1]),
        round(sensor_bl[0]),
        round(sensor_bl[1]),
    )


    return r

def sub_image(image, center, theta, width, height):
	"""Extract a rectangle from the source image.
	
	image - source image
	center - (x,y) tuple for the centre point.
	theta - angle of rectangle.
	width, height - rectangle dimensions.
	"""
	
	if 45 < theta <= 90:
		theta = theta - 90
		width, height = height, width
		
	theta *= math.pi / 180 # convert to rad
	v_x = (math.cos(theta), math.sin(theta))
	v_y = (-math.sin(theta), math.cos(theta))
	s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
	s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
	mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])

	return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def extractSensorArea(img, corner_points, cells):
    
   
    x0 = corner_points[0][0][0] #tl
    y0 = corner_points[0][0][1]

    x1 = corner_points[0][1][0] #tr
    y1 = corner_points[0][1][1]

    x2 = corner_points[0][2][0] #br
    y2 = corner_points[0][2][1]

    x3 = corner_points[0][3][0] #bl
    y3 = corner_points[0][3][1]
  
    angle_x = math.atan2(y1-y0, x1-x0)
    angle_y = math.atan2(y3-y0, x3-x0)
  
    angle = angle_x

    r = calculateSensorRect(corner_points,cells)

    #image = cv.circle(img, (int(x0),int(y0)),radius=3, color=(0, 0, 255), thickness=-1)
    #image = cv.circle(img, (int(x1),int(y1)),radius=3, color=(0, 255, 0), thickness=-1)
    #image = cv.circle(img, (int(x2),int(y2)),radius=3, color=(255, 0, 0), thickness=-1)
    #image = cv.circle(img, (int(x3),int(y3)),radius=3, color=(0, 0, 0), thickness=-1)
    #cv.imshow("img", image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    sensor_w = math.sqrt( ((r[0]-r[2])**2)+((r[1]-r[3])**2) )
    sensor_h = math.sqrt( ((r[0]-r[6])**2)+((r[1]-r[7])**2) )

    min_x = min(min(min(r[0],r[2]),r[4]),r[6])
    min_y = min(min(min(r[1],r[3]),r[5]),r[7])
    max_x = max(max(max(r[0],r[2]),r[4]),r[6])
    max_y = max(max(max(r[1],r[3]),r[5]),r[7])

    center_x = min_x + (max_x-min_x)/2
    center_y = min_y + (max_y-min_y)/2

    #image = cv.circle(img, (min_x,min_y),radius=3, color=(0, 0, 255), thickness=-1)
    #image = cv.circle(img, (max_x,max_y),radius=3, color=(0, 0, 255), thickness=-1)
    #image = cv.circle(img, (int(center_x),int(center_y)),radius=3, color=(0, 0, 255), thickness=-1)

    #cv.imshow("img", image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    crop_img = subimage(img, (int(center_x),int(center_y)), angle, int(sensor_w), int(sensor_h))

    #cv.imshow("img", crop_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return crop_img

def subimage(image, center, theta, width, height):

   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta = math.degrees(theta)
   v_x = (math.cos(theta), math.sin(theta))
   v_y = (-math.sin(theta), math.cos(theta))
   s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
   s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

   mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

   return cv.warpAffine(image,mapping,(width, height),flags=cv.WARP_INVERSE_MAP,borderMode=cv.BORDER_REPLICATE)


def make_2d_histogram(img):
    hsv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
    hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

    return hist

def makeTrainCodes(amount, sensor_color, percentage):

    for x in range(amount):
        
        content = randomword(random.randint(10, 20))

        img = qrcode.make(content)
        img.save("temp.jpg")   
        img = cv.imread("temp.jpg")

        det = cv.QRCodeDetector()
        data, corner_points, straight_qrcode = det.detectAndDecode(img)

        if corner_points is not None and straight_qrcode is not None:

            cells = len(straight_qrcode[0])
            r = calculateSensorRect(corner_points,cells)

            color = (sensor_color[0]*percentage, sensor_color[1]*percentage, sensor_color[2]*percentage)
            
            crop_img = extractSensorArea(img,corner_points,cells)
            crop_img[np.where((crop_img==[255,255,255]).all(axis=2))] = [sensor_color[0],sensor_color[1],sensor_color[2]]

            img[r[0]:r[1],r[2]:r[3]] = crop_img

            cv.imwrite("trainset/"+content+"_"+"+.jpg", crop_img)

 
            #cv.imshow("img", img)
            #cv.waitKey(0)
            #cv.destroyAllWindows()




def testRunSensorImage():
    purge("test_codes", "sensor")
    for dirpath, dirnames, files in os.walk('test_codes'):
        print(f'Found directory: {dirpath}')
        print("size,distance,sensor_expected,sensor_read,lux")
        for file_name in files:
            #print(file_name)
            if file_name.endswith(".DS_Store"):
                continue
            #####MAIN#######
            img = cv.imread("test_codes/"+file_name)
            det = cv.QRCodeDetector()
            data, points, straight_qrcode = det.detectAndDecode(img)
            #cv.imshow("img", img)

            #cv.waitKey(0)
            #cv.destroyAllWindows()
            
            if points is not None:
                #cells = len(straight_qrcode[0])
                #print(f"QRCode data:\n{data}")
                cells = int(file_name.split(".")[0].split("_")[0])

                crop_img = extractSensorArea(img,points,cells)

                two_dimensional_h = make_2d_histogram(crop_img)

                #plt.imshow(two_dimensional_h,interpolation = 'nearest')
                #plt.show()

                #cv.imshow("img", crop_img)

                #cv.waitKey(0)
                #cv.destroyAllWindows()
                
                #cv.imwrite('test_codes/sensor_'+file_name, crop_img)

                has_sensor = color_classification.classify(crop_img,3)

                if file_name.split(".")[0].split("_")[1] == "X":
                    distance = "X"
                else:
                    distance = int(file_name.split(".")[0].split("_")[1])

                if file_name.split(".")[0].split("_")[3] == "X":
                    lux = "X" 
                else:
                    lux = int(file_name.split(".")[0].split("_")[3])

                #filenaming size_distance_sensor%
                cells = file_name.split(".")[0].split("_")[0]
                expected_sensdor = file_name.split(".")[0].split("_")[2]
                
                size = int(file_name.split(".")[0].split("_")[0])
                
                print(str(cells)+";"+str(distance)+";"+str(expected_sensdor)+";"+str(has_sensor)+";"+str(lux))

                #cv.destroyAllWindows()
            #else:
                #print("FAILED")
#####MAIN#######

#makeTrainCodes(10000)
testRunSensorImage()
