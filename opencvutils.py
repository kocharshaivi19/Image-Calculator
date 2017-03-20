import numpy
import cv2
import sys
import os

def captch_ex(img):
    '''
    For any input image, it extracts the equation blob

    Input Parameters : 
    img : Input image

    Output Parameters : 
    cropped : Output equation image
    '''
    new_img = cv2.adaptiveThreshold(img,255,1,1,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3 , 3))
    dilated = cv2.dilate(new_img,kernel,iterations = 15) 
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cropped = np.array([])
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

        cropped = img[y :y +  h , x : x + w]
        cv2.imshow("cropped", cropped)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return cropped

def skewness(img):
    '''
    Detects rotation angle using HoughLines, Rotates the image based on Affine Transformation

    Input Parameters : 
    img : Input Image

    Output Image : Deskewed Image
    '''
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    print img.shape[1]
    print img.shape
    minLineLength=img.shape[1]/5
    lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/500, threshold=10,
                            lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)
    print lines
    
    angle = 0.0
    if lines != None:
        a,b,c = lines.shape
        print lines.shape
        for i in range(a):
            for j in range(b):
                print lines[i][j]
                angle = angle + math.atan2((lines[i][j][3] - lines[i][j][1]), 
                                            (lines[i][j][2] - lines[i][j][0]))
                print angle
        print angle
        angle = angle / (a*b);
        if angle != 0.0:
            center=tuple(np.array(img.shape[0:2])/2)
            rot_mat = cv2.getRotationMatrix2D(center,angle*180/cv2.cv.CV_PI,1.0)
            rotated = cv2.warpAffine(img, rot_mat, img.shape[0:2],flags=cv2.INTER_CUBIC)
        return rotated
    else:
        return []

def equationextractor(olddataset, equationdataset):
    '''
    Basic Operations on Image like Resizing to a constant size, 
    Blob detection, Deskewing the equation image, and storing the result Image

    Input Parameter : 
    olddataset : Path to original dataset
    equationdataset : Path to modified dataset 
    '''
    for filename in os.listdir(olddataset):
        if filename.endswith((".jpeg", ".png", "jpg")):
            print os.path.join(olddataset, filename)
            img = cv2.imread(os.path.join(olddataset, filename))            
            resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
            resized = cv2.bitwise_not(resized)
            
            skew_img = skewness(resized)
            
            if skew_img != []:
                cropped_img = self.captch_ex(skew_img)
                if not os.path.exists(equationdataset):
                    os.makedirs(equationdataset)
                s = os.path.join(equationdataset, filename)
                cv2.imwrite(s , cropped_img)

def digitextractor(equationdataset, digitdataset):
    '''
    Extract the digits and operators.
    It counts the nonzero pixels and segments each blob based on certain thresholding.
    It saves the blob into folders specific for each equation

    Input Parameter :
    equationdataset : path that reads the segmented equations from the image
    digitdataset : path to store the extracted blobs equation wise
    '''
    for filename in os.listdir(equationdataset):
        if filename.endswith((".jpeg", ".png", ".jpg")):
            print os.path.join(equationdataset, filename)
            img = cv2.imread(os.path.join(equationdataset, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, inv_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            col_arr = []
            
            for c in range(inv_img.shape[1]):
                col_arr.append(np.count_nonzero(inv_img[:, c]))
            print col_arr
            first = -1
            sec = -1
            index = 0
            for idx in range(col_arr-1):
                if col_arr[idx] < 10 and col_arr[idx+1] > 10 and first == -1:
                    first = idx
                elif col_arr[idx] > 10 and col_arr[idx+1] < 10 and sec == -1:
                    sec = idx + 1
                elif first != -1 and sec != -1:
                    cropped = img[0:img.shape[0], first : sec]
                    resized = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

                    bg = np.zeros(shape=(28,28))
                    roi = bg[4:18, 4:18]
                    r,mask = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    res_fg = cv2.bitwise_and(resized, resized, mask=mask)

                    bg = cv2.add(roi_bg, res_fg)
                    eq_folder = os.path.join(digitdataset, os.path.splittext(filename)[0])
                    if not os.path.exists(eq_folder):
                        os.makedirs(eq_folder)
                    cv2.imwrite(eq_folder + "/" + str(index) + '.jpg', bg);
                    first = -1
                    sec = -1
                    index = index + 1
                else:
                    continue