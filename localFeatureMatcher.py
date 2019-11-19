import numpy as np
import time
import cv2

# Determine if the sifted images match using the FLANN library for the Nearest Neighbor method
# (incorporated into openCV). Prints resulting marked-up image for matches and run-time for comparison
# to Brute Force method
def FLANNMethod(descriptors, keyPoints, runTimes, folder, grayscale):
    # Fast Library For Approximating Nearest Neighbors (FLANN) method
    method = 0
    beginning = time.time()
    index = dict(algorithm = 0, trees = 5)
    search = dict(checks = 50)
    fbm = cv2.FlannBasedMatcher(index, search)
    matchedPoints = fbm.knnMatch(descriptors[0],descriptors[1], k =2)

    goodMatchMask = filterMatches(matchedPoints, method) 

    featureMatchedPic = cv2.drawMatchesKnn(grayscale[0],keyPoints[0],grayscale[1],keyPoints[1],matchedPoints,None,matchesMask=goodMatchMask,flags=0,matchColor=(0,255,0),singlePointColor=(200,100,0))
    finish = time.time()
    runTimes.append(finish-beginning)
    imgFile="{}/flannMatcher.png".format(folder)
    cv2.imwrite(imgFile,featureMatchedPic)

    print("Total FLANN Method Time: {} seconds".format("%.3f"%runTimes[0]))

# Determine if the sifted images match using the Brute force method. Prints resulting
# marked-up image for matches and run-time for comparison to Brute Force method
def bruteForceMethod(descriptors, keyPoints, runTimes, folder, grayscale):
    method = 1
    beginning = time.time()
    bf = cv2.BFMatcher()
    matchedPoints = bf.knnMatch(descriptors[0],descriptors[1],k=2)

    # Filter matches for the better ones
    goodMatches = filterMatches(matchedPoints, method)

    featureMatchedPic = cv2.drawMatchesKnn(grayscale[0],keyPoints[0], grayscale[1], keyPoints[1],goodMatches,None,flags = 2)
    finish = time.time()
    runTimes.append(finish-beginning)
    imgFile = "{}/bruteForceMatch.png".format(folder)
    cv2.imwrite(imgFile, featureMatchedPic)
    
    print("Total Brute Force Method Time: {} seconds".format("%.3f"%runTimes[1]))

# Filters through the matched points found in the two compared the images
# and returns the stronger-coorelated matches    
def filterMatches(unfilteredMatches, method):
    if(method == 0):
        goodMatches = [[0,0] for x in range (len(unfilteredMatches))]
        for i,(j,k) in enumerate(unfilteredMatches):
            if j.distance < 0.7*k.distance:
                    goodMatches[i]=[1,0]
    else:
        goodMatches = []
        for i,j in unfilteredMatches:
            if i.distance < 0.75*j.distance:
                goodMatches.append([i])
    return goodMatches

# Uses Harris Corner Detection to extract corners and infer features of an image
def analyzePics(images, folder):
    index = 1
    runTimes = []
    keyPoints = []
    grayscale = []
    dscrptrs = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image in images:
    # Part 1: Using Harris Corner Detection
        # Extract Gray-Scaled Image
        pic = cv2.imread(image,0) 
        grayscale.append(pic)
        floatPic = np.float32(pic)
        # rgb Image
        rgbPic = cv2.imread(image) 
        harrisPic = rgbPic

        dst = cv2.cornerHarris(floatPic,2,5,0.07)
        dst = cv2.dilate(dst,None)
        harrisPic[dst>0.01*dst.max()]=[0,0,255]
        figureName = "{}/harrisImg{}.png".format(folder,index)
        cv2.imwrite(figureName,harrisPic)

    # Part 2: Distill features
        # Sift
        keyPoint,dscrptr = sift.detectAndCompute(pic, None)
        dscrptrs.append(dscrptr)
        keyPoints.append(keyPoint)
        
        siftedPic = cv2.drawKeypoints(pic,keyPoint,rgbPic,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        figureName = "{}/siftedImg{}.png".format(folder,index)
        cv2.imwrite(figureName,siftedPic)
        
        index+=1

    # Step 3: Determine if sifted images match using FLANN and brute force techniques
    FLANNMethod(dscrptrs, keyPoints, runTimes, folder, grayscale)
    bruteForceMethod(dscrptrs, keyPoints, runTimes, folder, grayscale)
    
# Extracts images from data folder and passes the to analyzePics to compare the 
# images and find matched points
def progDriver():
    storageFold="figures"

    notreDameFold="{}/notreDameFigs".format(storageFold)
    notreImg1="./data/Notre Dame/4191453057_c86028ce1f_o.jpg"
    notreImg2="./data/Notre Dame/921919841_a30df938f2_o.jpg"
    notreImgs = [notreImg1, notreImg2]
    analyzePics(notreImgs, notreDameFold)

    rushmoreFold="{}/rushmoreFigs".format(storageFold)
    pantheonImg1 = "./data/Rushmore/mountRushmore1.jpg"
    pantheonImg2 = "./data/Rushmore/mountRushmore2.jpg"
    pantheonImgs = [pantheonImg1,pantheonImg2]
    analyzePics(pantheonImgs, rushmoreFold)

progDriver()
