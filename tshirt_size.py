import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import imutils
from imutils import perspective 
from imutils import contours
from scipy.spatial import distance

debug = False

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Image resize code https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# PixelPerMetric code referrenced from https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized, dim
 
# Get T-Shirt Size for a given measurement.
def giveSizeAccordingToMeasurement(measurement):
    size = ''
    min = 1000
    sizechart = {'48':'S', '53':'M', '58': 'L', '63':'XL', '68':'XXL', '73':'XXXL'}
    for key in sizechart.keys():
        diff = abs(measurement - int(key))
        if diff < min:
            size = sizechart[key]
            min = diff
    return size

# get Dilated and Eroded image for the input Image.
def getDilatedErodedImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edged = cv2.Canny(gray, 100, 100)
    kernel = np.ones((3, 3), np.uint8)
    edg_dil = cv2.dilate(edged, kernel, iterations=3)
    edg_er = cv2.erode(edg_dil, kernel, iterations=3)
    return edg_er

# Get pixel per metric ratio, here we have used cm as metric
def getPixelPerMetric(noteImage, orientation):
    orig = noteImage.copy()
    pixelsPerMetric = None
    noteErroded = getDilatedErodedImage(noteImage)

    cnts = cv2.findContours(noteErroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    for count, c in enumerate(cnts):
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        
        topWidth = distance.euclidean(tl, tr)
        rightHeight = distance.euclidean(tr, br)
        bottomWidth = distance.euclidean(bl, br)
        leftHeight = distance.euclidean(bl, tl)

        if pixelsPerMetric is None:
            width = (topWidth + bottomWidth) / 2.0
            height = (rightHeight + leftHeight) / 2.0

            if orientation == 'Horizontal':
                pixelsPerMetric1 = width / 15.61
                pixelsPerMetric2 = height / 6.63
            else:
                pixelsPerMetric1 = width / 6.63
                pixelsPerMetric2 = height / 15.61
            
            pixelsPerMetric = (pixelsPerMetric1 + pixelsPerMetric2) / 2.0
            print("pixelPerMetric is {0} with new approach".format(pixelsPerMetric))

        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)
        # compute the Euclidean distance.distance between the midpoints
        dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
        print(dA, dB)

        print("DA is {0}".format(dA))
        print("DB is {0}".format(dB))
        print("PixelPerMatric is {0}".format(pixelsPerMetric))
        # compute the size of the object

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        print("for image dim A is {0}".format(dimA))
        print("for image dim B is {0}".format(dimB))

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}cm".format(dimB),
                    (int(tltrX-20), int(tltrY+20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimA),
                    (int(trbrX - 40), int(trbrY+20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)

        name = "Image_{0}".format(count)
        if debug == True:
            cv2.imshow(name, orig)
    return pixelsPerMetric

# Get Note Image depeding on the contour area.
def getNoteImage(errordedImg, inputImg):

    errod_img = errordedImg.copy()
    cnts = cv2.findContours(errod_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    
    for count, c in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 1000:
            continue

        print(cv2.contourArea(c))
        #if cv2.contourArea(c) > 1100 and cv2.contourArea(c) < 8000:
        if True:
            cx, cy, cw, ch = cv2.boundingRect(c)
            note = inputImg[cy-10:cy + ch+10, cx-10:cx + cw + 10]


            if debug == True:
                cv2.imshow("NoteImage", note)
            if cw > ch:
                orientation = 'Horizontal'
            else:
                orientation = 'Vertical'
    return note, orientation, cx, cy

# Get T-Shirt Image depensing on the contour area.
def getTshirtImage(errordedImg, inputImg):
    tshirt = None
    errod_img = errordedImg.copy()
    cnts1 = cv2.findContours(errod_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]
    (cnts1, _) = contours.sort_contours(cnts1, method="top-to-bottom")
    
    for count, c in enumerate(cnts1):
        if cv2.contourArea(c) > 50000:
            x, y, w, h = cv2.boundingRect(c)
            if x < 10 and y < 10:
                tshirt = inputImg[y:y + h + 10, x:x + w + 10]
            elif x > 10 and y < 10:
                tshirt = inputImg[y:y + h + 10, x-10:x + w + 10]
            elif x < 10 and y > 10:
                tshirt = inputImg[y-10:y + h + 10, x:x + w + 10]
            else:
                tshirt = inputImg[y-10:y + h + 10, x-10:x + w + 10]
            if debug == True: 
                cv2.imshow("croppedImage", tshirt)
    if tshirt is None:
        return None, 0, 0
    else:
        return tshirt, w, h

# Get size for an input Image.
def get_size(filename):
    try:
        tshirt = None
        ratio = None
        inputFilename, extension = filename.split('.')
        inputImg = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        inputImg, newDimensions = image_resize(inputImg, height=500)
        resizedFile = inputFilename + '_resized' + '.' + extension
        cv2.imwrite(resizedFile, inputImg)
        erroded = getDilatedErodedImage(inputImg)
        
        noteImg, orientation, leftx, lefty = getNoteImage(erroded, inputImg)
        ratio = getPixelPerMetric(noteImg, orientation)
        tshirt, tshirt_width, tshirt_height = getTshirtImage(erroded, inputImg)

        mid_x = newDimensions[0]/2.0
        mid_y = newDimensions[1]/2.0

        if debug == True:
            cv2.waitKey(0)

        if leftx < mid_x and lefty < mid_y and orientation == 'Horizontal':
            tshirt_orientation = '0'
        elif leftx > mid_x and lefty < mid_y and orientation == 'Vertical':
            tshirt_orientation = '90'
        elif leftx < mid_x and lefty > mid_y and orientation == 'Horizontal':
            tshirt_orientation = '180'
        else:
            tshirt_orientation = '270'

        print(tshirt_orientation)
        
        if tshirt is not None:
            blur = cv2.GaussianBlur(tshirt, (5, 5), 0)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            cv2.imwrite("BlurImage.jpg", blur)

            edge = cv2.Canny(blur, 100, 100)
            midpoint_x = int(tshirt_width / 2)
            midpoint_y = int(tshirt_height / 2)

            if debug == True:
                cv2.imshow("EdgesDetected", edge)
            cv2.imwrite("EdgesDetected.jpg", edge)

            corners = cv2.goodFeaturesToTrack(edge, 18, 0.2, 20)
            corners = np.int0(corners)

            quadrant1 = []
            quadrant2 = []
            quadrant3 = []
            quadrant4 = []

            for corner in corners:
                x, y = corner.ravel()
                if tshirt_orientation == '0':
                    if x > midpoint_x and y > midpoint_y:
                        quadrant4.append((x, y))
                    elif x < midpoint_x and y > midpoint_y:
                        quadrant3.append((x, y))
                    elif x > midpoint_x and y < midpoint_y:
                        quadrant1.append((x,y))
                    else:
                        quadrant2.append((x,y))
                
                if tshirt_orientation == '90':
                    if x < midpoint_x and y < midpoint_y:
                        quadrant3.append((x, y))
                    elif x < midpoint_x and y > midpoint_y:
                        quadrant4.append((x, y))
                    elif x > midpoint_x and y > midpoint_y:
                        quadrant1.append((x,y))
                    else:
                        quadrant2.append((x,y))
                
                if tshirt_orientation == '180':
                    if x > midpoint_x and y < midpoint_y:
                        quadrant3.append((x, y))
                    elif x < midpoint_x and y < midpoint_y:
                        quadrant4.append((x, y))
                    elif x < midpoint_x and y > midpoint_y:
                        quadrant1.append((x,y))
                    else:
                        quadrant2.append((x,y))

                if tshirt_orientation == '270':
                    if x > midpoint_x and y > midpoint_y:
                        quadrant3.append((x, y))
                    elif x > midpoint_x and y < midpoint_y:
                        quadrant4.append((x, y))
                    elif x < midpoint_x and y < midpoint_y:
                        quadrant1.append((x,y))
                    else:
                        quadrant2.append((x,y))

            print(quadrant1)
            print(quadrant2)
            print (quadrant3)
            print (quadrant4)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(edge, (x, y), 3, 255, -1)
        
            if debug == True:
                cv2.imshow("Corners", edge)
            cv2.imwrite("CornersDetected.jpg", edge)

            # sort quadrant 3 points from left to right
            # sort quadrant 4 points from right to left
            if tshirt_orientation == '0':
                quadrant1.sort(key=lambda x: x[1], reverse=True)
                quadrant2.sort(key=lambda x: x[1], reverse=True)
                quadrant3.sort(key=lambda x: x[0])
                quadrant4.sort(key=lambda x: x[0], reverse=True)
                quadrant1 = quadrant1[0:2]
                quadrant2 = quadrant2[0:2]
                quadrant1.sort(key=lambda x:x[0])
                quadrant2.sort(key=lambda x:x[0], reverse=True)
            
            if tshirt_orientation == '90':
                quadrant1.sort(key=lambda x: x[0])
                quadrant2.sort(key=lambda x: x[0])
                quadrant3.sort(key=lambda x: x[1])
                quadrant4.sort(key=lambda x: x[1], reverse=True)
                quadrant1 = quadrant1[0:2]
                quadrant2 = quadrant2[0:2]
                quadrant1.sort(key=lambda x: x[1])
                quadrant2.sort(key=lambda x: x[1], reverse=True)
            
            if tshirt_orientation == '180':
                quadrant1.sort(key=lambda x: x[1])
                quadrant2.sort(key=lambda x: x[1])
                quadrant3.sort(key=lambda x: x[0], reverse=True)
                quadrant4.sort(key=lambda x: x[0])
                quadrant1 = quadrant1[0:2]
                quadrant2 = quadrant2[0:2]
                quadrant1.sort(key=lambda x: x[0], reverse=True)
                quadrant2.sort(key=lambda x: x[0])

            if tshirt_orientation == '270':
                quadrant1.sort(key=lambda x: x[0], reverse=True)
                quadrant2.sort(key=lambda x: x[0], reverse=True)
                quadrant3.sort(key=lambda x: x[1], reverse=True)
                quadrant4.sort(key=lambda x: x[1])
                quadrant1 = quadrant1[0:2]
                quadrant2 = quadrant2[0:2]
                quadrant1.sort(key=lambda x: x[1], reverse=True)
                quadrant2.sort(key=lambda x: x[1])

            
            x1, y1 = quadrant1[0]
            x2, y2 = quadrant2[0]
            x3, y3 = quadrant3[0]
            x4, y4 = quadrant4[0]

            print(quadrant1)
            print(quadrant2)
            print(quadrant3)
            print(quadrant4)

            width1 = distance.euclidean((x1,y1),(x2,y2))
            width1 = width1 / ratio
            print("distance between ({0},{1}) and ({2},{3}) points is {4}".format(x1, y1, x2, y2, width1))

            width2 = distance.euclidean((x3, y3),(x4, y4))
            width2 = width2 / ratio
            print("distance between ({0},{1}) and ({2},{3}) points is {4}".format(x3, y3, x4, y4, width2))
            
            # if chest is smaller than the waist then consider chest else deduct 2 from waist
            if width1 < width2:
                real = width1 # take chest size
            else:
                real = width2 - 1 # adjust for expansion in ending
            
            print("real distance is {0}".format(real))
        
            if debug == True:
                cv2.waitKey(0)
            return giveSizeAccordingToMeasurement(real)
        else:
            return 'Error, T-shirt not detected!'
    except Exception as e:
        return 'Error is {0}'.format(str(e))
        

if __name__ == "__main__":
    debug = False
    print('Input Image T- shirt size is {0}'.format(get_size("images/my.jpg")))










