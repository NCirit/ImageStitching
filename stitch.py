# import the necessary packages
import numpy as np
import imutils
import cv2
from PIL import Image
class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)
    def max(self, num1, num2):
        return num1 if num1 > num2 else num2

    def min(self, num1, num2):
        return num1 if num1 < num2 else num2

    def getMatchedPoints(self, kpsA, kpsB, featuresA, featuresB, ratio):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
           # ensure the distance is within a certain ratio of each
           # other (i.e. Lowe’s ratio test)
           if len(m) == 2 and m[0].distance < m[1].distance * ratio:
              matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 3:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])
            return (ptsA, ptsB)
        return None

    def getRigidTransform(self,img_paths):
        first = cv2.imread(img_paths[0])
        (kpsFirst, featuresFirst) = self.detectAndDescribe(first)
        H = []
        for i in range(1,len(img_paths)):
            second = cv2.imread(img_paths[i])
            (kpsSecond, featuresSecond) = self.detectAndDescribe(second)
            (ptsFirst ,ptsSecond) = self.getMatchedPoints(kpsFirst, kpsSecond, featuresFirst, featuresSecond, 0.75)

            M = cv2.estimateRigidTransform(ptsFirst,ptsSecond, False, 50, 0.5, 3)

            H.append(M)
            kpsFirst = kpsSecond
            featuresFirst = featuresSecond
        return H



    def calcOffsets(self, img_paths):
        offsets = []

        Hs = self.getRigidTransform(img_paths)
        for i in range(len(img_paths) - 1):
            #((matches, H, status), shapeA) = self.getHomography(img_paths[i], img_paths[i+1])
            H = Hs[i]
            
            if H is None:
                return None

            offset = np.dot(H, [0, 0, -1])
            #print(np.dot(Hs[i], [0, 0, 1]))
            #print(offset)
            #offset = offset / offset[-1]
            offset = offset.astype(np.int)
            im = Image.open(img_paths[i])
            shapeA = im.size[::-1]
            
            """
            center = [0,0]
            if offset[0] < 0:
                center[0] += abs(offset[0])
            if offset[1] < 0:
                center[1] += abs(offset[1])
            StartX = center[0] + offset[0]
            """
            # (0,0) point is assumed to be the upper right corner of the left image
            offset[0] = offset[0] - shapeA[1]
            offset[1] *= -1
            offsets.append(offset)
        return offsets

    def saveOffsetToFile(self, offsets):
        fl = open("offsets.txt", "w")
        fl.write("imgID,imgID,offsetX,offsetY\n")
        for i in range(len(offsets)):
            fl.write(str(i) + "," + str(i + 1) + ",")
            fl.write(str(offsets[i][0]) + "," + str(offsets[i][1]) + "\n")
        
        fl.close()
    def multiStitch(self, img_paths):
        offsets = self.calcOffsets(img_paths)
        print(offsets)
        if offsets is None:
            print("Couldn't find any realationship between images")
            return None

        #self.getAffine(img_paths)
        self.saveOffsetToFile(offsets)

        imgs = []
        for i in img_paths:
            imgs.append(cv2.imread(i))

        resultShape = list(imgs[0].shape)
        positions = []
        offsetYStart = offsets[0][1]
        offsetYEnd = -imgs[0].shape[0]

        offsetYAccumulator = offsets[0][1]

        for i in range(1,len(imgs)):
            # x ekseni
            offsetYEnd = self.max(offsetYEnd, offsetYAccumulator - imgs[i].shape[0])
            # y ekseni
            if i < len(offsets):
                offsetYStart = self.min(offsetYStart, offsetYStart + offsets[i][1])   
                offsetYAccumulator += offsets[i][1]

            resultShape[1] += imgs[i].shape[1] + offsets[i - 1][0]

        # This conditions is true only if there is no intersection 
        if offsetYStart < offsetYEnd:
            print("No intersecting area")
            return None
            
        resultShape[0] = abs(offsetYEnd) - abs(offsetYStart)

        result = np.zeros(shape = resultShape, dtype=np.uint8)
        result[:,:imgs[0].shape[1]] = imgs[0][abs(offsetYStart):abs(offsetYEnd),:]
        

        endpointX = imgs[0].shape[1]
        print(endpointX)
        endpointY = abs(offsetYStart)
        for i in range(1,len(imgs)):
            endpointX += offsets[i - 1][0]
            offsets[i - 1][1] += endpointY
            cv2.imshow("asd",imgs[i])
            result[:, endpointX:imgs[i].shape[1] + endpointX] = imgs[i][offsets[i - 1][1]:offsets[i - 1][1]- abs(offsetYStart) + abs(offsetYEnd),:]
            endpointY = offsets[i - 1][1]
            endpointX += imgs[i].shape[1]
        return result
        
        

    def getHomography(self, imgA_path, imgB_path):
        imgA = cv2.imread(imgA_path)
        imgB = cv2.imread(imgB_path)
        (kpsA, featuresA) = self.detectAndDescribe(imgA)
        (kpsB, featuresB) = self.detectAndDescribe(imgB)

        M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio=0.75, reprojThresh=4.0)
        
        # returns None if there is no match between images
        if M is None:
            return None
        return (M, imgA.shape)

    def stitch(self, img_paths, ratio=0.75, reprojThresh=4.0,showMatches=False):
        ((matches, H, status), shape0) = self.getHomography(img_paths[0], img_paths[1])
        

        imageA = cv2.imread(img_paths[1])
        imageB = cv2.imread(img_paths[0])

        offset = np.dot(H, [0,0,1])
        offset = offset / offset[-1]

        center = [0,0]
        if offset[0] < 0:
            center[0] += abs(int(offset[0]))
        if offset[1] < 0:
            center[1] += abs(int(offset[1]))

        result = np.zeros(shape=(center[1] + self.max(int(offset[1]) + imageA.shape[0],imageB.shape[0]), center[0] + self.max(int(offset[0]) + imageA.shape[1], imageB.shape[1]), 3), dtype=np.uint8)
        # according to left images upper left corner point is (0,0)
        #print("Offset X:" + str(int(offset[0])))
        #print("Offset Y:" + str(int(offset[1])))
        result[center[1]:imageB.shape[0] + center[1], center[0]:imageB.shape[1]+center[0]] = imageB
        result[center[1] + int(offset[1]):imageA.shape[0] + int(offset[1]) + center[1], center[0] + int(offset[0]):imageA.shape[1] + int(offset[0])+center[0]] = imageA
        

        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return (result, None)

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SURF_create()
            #descriptor = cv2.ORB_create(nfeatures=1500)
            (kps, features) = descriptor.detectAndCompute(image, None)
            # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SURF")
            kps = detector.detect(gray)
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SURF")
            (kps, features) = extractor.compute(gray, kps)
            # convert the keypoints from KeyPoint objects to NumPy
            # arrays
            kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
           # ensure the distance is within a certain ratio of each
           # other (i.e. Lowe’s ratio test)
           if len(m) == 2 and m[0].distance < m[1].distance * ratio:
              matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
           # only process the match if the keypoint was successfully
           # matched
           if s == 1:
             # draw the match
             ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
             ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
             cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis
