import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

print 'OpenCV Version:'
print cv2.__version__

plot = 1

######################################### functions ###################################################

def plot_img(img, cmap=None):
	if plot == 1:
		plt.imshow(img, cmap)
		plt.show()
	return

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    plot_img(cv2.cvtColor(img, cv2.CV_32S))
    return 

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plot_img(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

def get_file_list(path):
	return [f for f in listdir(path) if isfile(join(path, f))]

def sift(imgPath):
	#read image
	img = cv2.imread(imgPath)

	#plot image
	show_rgb_img(img)

	#convert to gray color
	img_gray = to_gray(img)

	#plot image in gray color
	plot_img(img_gray, cmap='gray')

	# generate SIFT keypoints and descriptors
	keypoints, descriptors = gen_sift_features(img_gray)

	#show SIFT features
	show_sift_features(img_gray, img, keypoints);
	return keypoints, descriptors

def test_sift_algorithm():
	# Reading images
	octo_front = cv2.imread('dataset/Octopus_Far_Front.jpg')
	octo_offset = cv2.imread('dataset/Octopus_Far_Offset.jpg')

	# Show images
	show_rgb_img(octo_front)
	show_rgb_img(octo_offset)

	# Convert to gray color
	octo_front_gray = to_gray(octo_front)
	octo_offset_gray = to_gray(octo_offset)

	# Plot image in gray color
	plot_img(octo_front_gray, cmap='gray')

	# generate SIFT keypoints and descriptors
	octo_front_kp, octo_front_desc = gen_sift_features(octo_front_gray)
	octo_offset_kp, octo_offset_desc = gen_sift_features(octo_offset_gray)

	#Show features calculated using SIFT
	print 'Here are what our SIFT features look like for the front-view octopus image:'
	show_sift_features(octo_front_gray, octo_front, octo_front_kp);

	# Visualize how the SIFT features matches with the two images
	# create a BFMatcher object which will match up the SIFT features
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	matches = bf.match(octo_front_desc, octo_offset_desc)

	# Sort the matches in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# draw the top N matches
	N_MATCHES = 100

	match_img = cv2.drawMatches(octo_front, octo_front_kp,
                            	octo_offset, octo_offset_kp,
                            	matches[:N_MATCHES], octo_offset.copy(), flags=0)

	plt.figure(figsize=(12,6))
	plot_img(match_img);
	return

########################################## Main Program ################################################

datasetPath = 'dataset'
outputDataSet = 'outdata.txt'

#test_sift_algorithm()

outputFile = open(outputDataSet, 'a')

# Get all files
filesDataSet = get_file_list(datasetPath)

# Iterate the files and get the keypoints and descriptors
for file in filesDataSet:
	
	imgPath = datasetPath + '/' + file
	print imgPath

	# Use SIFT algorithm
	keyp, desc = sift(imgPath)

	# write file name
	outputFile.write("FILE: " + imgPath + '\n')
	# write keypoints
	outputFile.write(" ".join(str(elm) for elm in keyp))
	outputFile.write('\n')
	# write descriptors
	outputFile.write(" ".join(str(elm) for elm in desc))
	outputFile.write('\n')

# Close output file
outputFile.close()

print 'END'