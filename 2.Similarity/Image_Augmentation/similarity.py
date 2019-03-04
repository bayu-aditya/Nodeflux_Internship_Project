# Author : Bayu Aditya

import cv2
import numpy as np
import matplotlib.pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out

def compareSIFT(img1, img2):
    #img1 = cv2.imread(filename1)          # queryImage
    #img2 = cv2.imread(filename2)          # trainImage
    
    # Initiate SIFT detector
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    clusters = np.array([des1])
    bf.add(clusters)

    # Train: Does nothing for BruteForceMatcher though.
    bf.train()

    matches = bf.match(des2)
    matches = sorted(matches, key = lambda x:x.distance)

    #matches = bf.match(des1,des2)
    #matches = sorted(matches, key=lambda val: val.distance)

    #img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])
    #plt.imshow(img3)
    #plt.show()
    return matches

def compareSURF(img1, img2):
    #img1 = cv2.imread(filename1)          # queryImage
    #img2 = cv2.imread(filename2)          # trainImage
    
    # Initiate SIFT detector
    #sift = cv2.SIFT()
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=50000, upright=True, extended=True)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    #matches = bf.match(des1,des2)

    clusters = np.array([des1])
    bf.add(clusters)
    bf.train()
    matches = bf.match(des2)
    #matches = sorted(matches, key=lambda val: val.distance)

    #img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])
    #plt.imshow(img3)
    #plt.show()
    return matches

def compareORB(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)          # queryImage
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)          # trainImage
    
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    #matches = sorted(matches, key=lambda val: val.distance)

    #img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])
    #plt.imshow(img3)
    #plt.show()
    return matches