# import the necessary packages
from stitch import Stitcher
import argparse
import imutils
import cv2
from imutils import paths
import numpy as np
# construct the argument parse and parse the arguments

#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "-first", required=True, help="path to the first image")
#ap.add_argument("-s", "-second", required=True, help="path to the second image")
#args = vars(ap.parse_args())

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
        help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
        help="path to the output image")
args = vars(ap.parse_args())

img_paths = list(paths.list_images(args["images"]))


# load the two images and resize them to have a width of 400 pixels
# (for faster processing)

print(img_paths)
stitcher = Stitcher()


pivot = stitcher.multiStitch(img_paths)
"""
(temp, vis) = stitcher.stitch(img_paths)
pivot = temp
offsets = stitcher.calcOffsets(img_paths)
print(offsets)

imgA = cv2.imread(img_paths[0])
imgB = cv2.imread(img_paths[1])
print("Shape A: ", imgA.shape)
print("Shape B:", imgB.shape)
testOffset = np.zeros(shape = (stitcher.max(imgA.shape[0], imgB.shape[0] - offsets[0][1]), imgA.shape[1] + offsets[0][0] + imgB.shape[1], 3), dtype=np.uint8)
print("Shape TestOffset:", testOffset.shape)
testOffset[:imgA.shape[0], :imgA.shape[1]] = imgA
testOffset[:imgB.shape[0] - offsets[0][1], offsets[0][0] + imgA.shape[1]:] = imgB[offsets[0][1]:,:]
cv2.imwrite("out_coord.jpg", testOffset)
cv2.imshow("test offset",testOffset)
"""

cv2.imwrite(args["output"], pivot) 
cv2.imshow("PIVOT", pivot)
       
"""imageA = cv2.imread(args["f"])
imageB = cv2.imread(args["s"])
#imageA = imutils.resize(imageA, width=400)
#imageB = imutils.resize(imageB, width=400)
# stitch the images together to create a panorama
# show the images
cv2.imwrite("out.jpg",result)
print(result.shape)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
"""
#if vis is not None:
#    cv2.imshow("Keypoint Matches", vis)
#cv2.imshow("Result", result)

cv2.waitKey(0)


