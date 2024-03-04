import numpy as np
import cv2
#from skimage.transform import warp

#PREPROCESSING FUNCTIONS (taken from Ternes et. al. implementation code)
# These are used in the process_crc_wsi.py and proces_crc_tma.py scripts after cells are cropped and the 
# background is zeroed out. These are the same functions used to generate the Breast Cancer dataset from Ternes et. al.

#rotate 
def rotate_image(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    #new_image = warp(image, inverse_map=np.vstack([rot_mat, [0,0,1]]), output_shape=(col,row), preserve_range=True)
    return new_image

#flip
def flip_mask(image, mask, CROP_SIZE):
    #Identify quadrant
    up_left = np.mean(image[:int(CROP_SIZE/2),:int(CROP_SIZE/2),0][image[:int(CROP_SIZE/2),:int(CROP_SIZE/2),0]>0])
    down_left = np.mean(image[int(CROP_SIZE/2):,:int(CROP_SIZE/2),0][image[int(CROP_SIZE/2):,:int(CROP_SIZE/2),0]>0])
    up_right = np.mean(image[:int(CROP_SIZE/2),int(CROP_SIZE/2):,0][image[:int(CROP_SIZE/2),int(CROP_SIZE/2):,0]>0])
    down_right = np.mean(image[int(CROP_SIZE/2):,int(CROP_SIZE/2):,0][image[int(CROP_SIZE/2):,int(CROP_SIZE/2):,0]>0])
    
    vec = [up_left, down_left, up_right, down_right]
    index = np.argmax(vec)
    
    if index == 0:
        FlippedImage = image
        FlippedMask = mask
    elif index == 1:
        FlippedImage = np.flipud(image)
        FlippedMask = np.flipud(mask)
    elif index == 2:
        FlippedImage = np.fliplr(image)
        FlippedMask = np.fliplr(mask)
    elif index == 3:
        FlippedImage = np.fliplr(image)
        FlippedImage = np.flipud(FlippedImage)
        FlippedMask = np.fliplr(mask)
        FlippedMask = np.flipud(FlippedMask)
    
    return FlippedImage, FlippedMask
