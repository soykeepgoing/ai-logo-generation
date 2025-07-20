from PIL import Image
import cv2 
import skimage.metrics
import numpy as np
import math 
from skimage.measure import regionprops

def extract_symmetry(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    image = pil_image.convert('L')

    # Convert the input image to a numpy array
    image_np = np.array(image)
    flipped_image_np_list = []
    
    # Flip the original image vertically (top to bottom)
    vertical_image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    vertical_image_np = np.array(vertical_image)
    flipped_image_np_list.append(vertical_image_np)
    
    # Flip the original image horizontally (left to right)
    horz_image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    horz_image_np = np.array(horz_image)
    flipped_image_np_list.append(horz_image_np)
    
    # Flip the original image diagonally (bottom-left to top-right)
    diag_image_np_1 = np.rot90(np.fliplr(image_np))
    flipped_image_np_list.append(diag_image_np_1)
    
    # Flip the original image diagonally (bottom-right to top-left)
    diag_image_np_2 = np.flip(image_np, axis=(0, 1))
    flipped_image_np_list.append(diag_image_np_2)

    # Initialize the maximum symmetry score (SSIM) to -1
    symmetry = -1
    for flipped_image_np in flipped_image_np_list:
        # Calculate the Structural Similarity Index (SSIM) between the original and flipped image
        mse = skimage.metrics.structural_similarity(image_np, flipped_image_np)
        
        # Update the symmetry score with the maximum SSIM value found
        symmetry = max(symmetry, mse)

    return round(symmetry,3)

def extract_roundness(image):
    
    # Convert the image to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny algorithm
    # threshold1 and threshold2 define the hysteresis thresholding values for Canny edge detection
    threshold1 = 50
    threshold2 = 150
    canny =  cv2.Canny(gray, threshold1, threshold2)

    # Create a binary image from the Canny edge output using thresholding
    thresh = cv2.threshold(canny, 200, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological closing (dilation followed by erosion) to remove small holes or gaps
    # Kernel size is set to (2,2), which defines the size of the structuring element used for closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = thresh.astype('uint8')

    # Find contours in the binary image, only considering the external contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Find the largest contour, assumed to be the object of interest
    big_contour = max(contours, key=cv2.contourArea)

    # Create a black background and draw the largest contour filled with white
    result = np.full_like(thresh, 0)
    cv2.drawContours(result, [big_contour], -1, 255, cv2.FILLED)

    # Calculate region properties of the detected shape
    r = regionprops(result)[0] # Measure the region of the detected shape
    # Compute roundness: Roundness is defined as (4 * π * Area) / (Perimeter^2)
    roundness = (4 * math.pi * r.area) / (r.perimeter ** 2)

    return round(roundness,3)

def extract_repetition(image):
    # image load 
    blur_image_low = cv2.GaussianBlur(image, (1,1),0)
    blur_image_high = cv2.GaussianBlur(image, (19,19),3)

    # calculate mean value of distance between two images 
    diff = np.mean((blur_image_low - blur_image_high)**2)
    
    return round(diff,3)

def sort_xy( x, y):
    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)
    mask = list(mask)

    x_sorted = [x[idx] for idx in mask]
    y_sorted = [y[idx] for idx in mask]

    return x_sorted, y_sorted

def extract_proportion(image):   
    # get binary image 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    im_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    im_th_inv = cv2.bitwise_not(im_th)

    # find contour 
    contours = cv2.findContours(im_th_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # merge all contours point and generate list 
    list_of_pts = [] 
    for ctr in contours:
        ctr_list = ctr.tolist()
        list_of_pts += [pt[0] for pt in ctr_list]

    # get the clockwise sorting of x, y 
    # calculate clear edges of the icon 
    x,y = list(zip(*list_of_pts))
    x_sorted, y_sorted = sort_xy(x,y)

    list_of_pts_sorted = [[x_sorted[i], y_sorted[i]] for i in range(len(x_sorted))]

    ctr = np.array(list_of_pts_sorted).reshape((-1,1,2)).astype(np.int32)

    x,y,w,h = cv2.boundingRect(ctr)

    proportion = float(w)/h
    return round(proportion,3)

def extract_elaborate(image):
    # image convert color into lab
    lab_image_np = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    w,h, _ = image.shape

    L, A, B = cv2.split(lab_image_np)

    # Calculate gradient 
    L_grads = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize = 1)
    A_grads = cv2.Sobel(A, cv2.CV_64F, 1, 0, ksize = 1)
    B_grads = cv2.Sobel(B, cv2.CV_64F, 1, 0, ksize = 1)

    grads = [[0 for _ in range(h)] for _ in range(w)]
    gradsum = 0 

    for i in range(h):
        L_grad = L_grads[i]
        A_grad = A_grads[i]
        B_grad = B_grads[i]
        for j in range(w):
            grads[i][j] = max(L_grad[j], A_grad[j], B_grad[j])
            gradsum += grads[i][j]

    gradsum /= (w * h)

    return round(gradsum,3)