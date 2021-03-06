#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
from plantcv import plantcv as pcv 
import numpy as np
import cv2


# In[10]:


class options:
    def __init__(self):
        self.image = "C:/Users/lacho/Downloads/16-00-00.jpg"
        self.debug = "plot"
        self.writeimg= False 
        self.result = "./morphology_tutorial_results2.json"
        self.outdir = "."

# Get options
args = options()

# Set debug to the global parameter 
pcv.params.debug = args.debug


# In[11]:


# Read image (sometimes you need to run this line twice to see the image) 

# Inputs:
#   filename - Image file to be read in 
#   mode - How to read in the image; either 'native' (default), 'rgb', 'gray', or 'csv'
img, path, filename = pcv.readimage(filename=args.image)


# In[12]:


#crop Image

from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)

pcv.params.debug = "plot"

# Crop image
crop_img = pcv.crop(img=img, x=400, y=20, h=100, w=100)


# In[13]:


# Check if this is a night image, for some of these dataset's images were captured
# at night, even if nothing is visible. To make sure that images are not taken at
# night we check that the image isn't mostly dark (0=black, 255=white).
# if it is a night image it throws a fatal error and stops the workflow.

if np.average(img) < 50:
    pcv.fatal_error("Night Image")
else:
    pass

# Normalize the white color so you can later
# compare color between images.

# Inputs:
#   img = image object, RGB color space
#   roi = region for white reference, if none uses the whole image,
#         otherwise (x position, y position, box width, box height)

# white balance image based on white toughspot
img1 = pcv.white_balance(crop_img, roi=(0,0,200,20))


# In[14]:


# Convert RGB to HSV and extract the saturation channel
# Then set threshold for saturation
s = pcv.rgb2gray_hsv(rgb_img=img1, channel='s')
s_thresh = pcv.threshold.binary(gray_img=s, threshold=130, max_value=255, object_type='dark')


# In[15]:


# Set Median Blur
#Input box size "ksize" 
s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=2)
s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=2)


# In[16]:


# Convert RGB to LAB and extract the blue channel
#Then threshold the image
b = pcv.rgb2gray_lab(rgb_img=img1, channel='b')
b_thresh = pcv.threshold.binary(gray_img=b, threshold=135, max_value=255, object_type='light')

#Setting threshold continued
b_cnt = pcv.threshold.binary(gray_img=b, threshold=135, max_value=255, object_type='light')


# In[18]:


#Join the blue and yellow binary images
bs = pcv.logical_and(bin_img1=s_mblur, bin_img2=b_cnt)

masked = pcv.apply_mask(img=img1, mask=bs, mask_color='white')

#identify objects
obj2 = id_objects,obj_hierarchy = pcv.find_objects(img=masked, mask=bs)

#Define Range of Intrest
# Inputs: 
    #   img - RGB or grayscale image to plot the ROI on 
    #   x - The x-coordinate of the upper left corner of the rectangle 
    #   y - The y-coordinate of the upper left corner of the rectangle 
    #   h - The height of the rectangle 
    #   w - The width of the rectangle 
roi1, roi_hierarchy= pcv.roi.rectangle(img=img1, x=40, y=20, h=20, w=20)


# In[19]:


# Decide which objects to keep

    # Inputs:
    #    img            = img to display kept objects
    #    roi_contour    = contour of roi, output from any ROI function
    #    roi_hierarchy  = contour of roi, output from any ROI function
    #    object_contour = contours of objects, output from pcv.find_objects function
    #    obj_hierarchy  = hierarchy of objects, output from pcv.find_objects function
    #    roi_type       = 'partial' (default, for partially inside), 'cutto', or 
    #    'largest' (keep only largest contour)
roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=crop_img, roi_contour=roi1, 
                                                                  roi_hierarchy=roi_hierarchy,
                                                                  object_contour=id_objects,
                                                                  obj_hierarchy=obj_hierarchy,
                                                                  roi_type='partial')

#mine
cropped_mask=kept_mask


# In[20]:


#[BREAK END]
#Crop the mask 
#cropped_mask = img[1:2000, 1:2530]

# Skeletonize the mask 
#%matplotlib notebook
# To enable the zoom feature to better see fine lines, uncomment the line above ^^ 

# Inputs:
#   mask = Binary image data
skeleton = pcv.morphology.skeletonize(mask=cropped_mask)


# In[31]:



# Prune the skeleton  
# Generally, skeletonized images will have barbs (this image is particularly ideal, 
# that's why it's the example image in the tutorial!), 
# representing the width, that need to get pruned off. 

# Inputs:
#   skel_img = Skeletonized image
#   size     = Size to get pruned off each branch
#   mask     = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
#img1, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=2, mask=cropped_mask)

pcv.params.line_thickness = 2 
segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=skeleton)
segmented_img2, obj = pcv.morphology.segment_skeleton(skel_img=skeleton, 
                                                      mask=cropped_mask)
pcv.params.line_thickness = 2 
leaf_obj, other_obj = pcv.morphology.segment_sort(skel_img=skeleton,
                                                  objects=obj)
leaf_obj, other_obj = pcv.morphology.segment_sort(skel_img=skeleton,
                                                  objects=obj,
                                                  mask=cropped_mask)


# In[32]:


import os
import cv2
import numpy as np
from skimage.segmentation import watershed
from plantcv.plantcv import fatal_error
from plantcv.plantcv import outputs
from plantcv.plantcv import color_palette
from plantcv.plantcv import params
from plantcv.plantcv import plot_image
from plantcv.plantcv import print_image


def fill_segments(mask, objects):
    """Fills masked segments from contours.
    Inputs:
    mask         = Binary image, single channel, object = 1 and background = 0
    objects      = List of contours
    Returns:
    filled_img   = Filled mask
    :param mask: numpy.ndarray
    :param object: list
    :return filled_img: numpy.ndarray
    """

    params.device += 1

    h,w = mask.shape
    markers = np.zeros((h,w))

    labels = np.arange(len(objects)) + 1
    for i,l in enumerate(labels):
        cv2.drawContours(markers, objects, i ,int(l) , 5)

    # Fill as a watershed segmentation from contours as markers
    filled_mask = watershed(mask==0, markers=markers,
                            mask=mask!=0,compactness=0)

    # Count area in pixels of each segment
    ids, counts = np.unique(filled_mask, return_counts=True)
    outputs.add_observation(variable='segment_area', trait='segment area',
                            method='plantcv.plantcv.morphology.fill_segments',
                            scale='pixels', datatype=list,
                            value=counts[1:].tolist(),
                            label=(ids[1:]-1).tolist())

    rgb_vals = color_palette(num=len(labels))
    filled_img = np.zeros((h,w,3), dtype=np.uint8)
    for l in labels:
        for ch in range(3):
            filled_img[:,:,ch][filled_mask==l] = rgb_vals[l-1][ch]

    if params.debug == 'print':
        print_image(filled_img, os.path.join(params.debug_outdir, str(params.device) + '_filled_img.png'))
    elif params.debug == 'plot':
        plot_image(filled_img)

    return filled_img


# In[35]:


#from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file),
# or "plot" (Jupyter Notebooks or X11)
#pcv.params.debug = "print"

filled_img = pcv.morphology.fill_segments(mask=cropped_mask, objects=obj, label="default")

# Access data stored out from fill_segments
segments_area = pcv.outputs.observations['default']['segment_area']['value']


# In[36]:


# Identify branch points   

# Inputs:
#   skel_img = Skeletonized image
#   mask     = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=cropped_mask)


# In[37]:


# Identify tip points   

# Inputs:
#   skel_img = Skeletonized image
#   mask     = (Optional) binary mask for debugging. If provided, debug 
#              image will be overlaid on the mask.
tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=cropped_mask)


# In[38]:


# Adjust line thickness with the global line thickness parameter (default = 5),
# and provide binary mask of the plant for debugging. NOTE: the objects and
# hierarchies returned will be exactly the same but the debugging image (segmented_img)
# will look different.
pcv.params.line_thickness = 2


# In[39]:


# Sort segments into primary (stem) objects and secondary (leaf) objects. 
# Downstream steps can be performed on just one class of objects at a time, 
# or all objects (output from segment_skeleton) 
  
# Inputs:
#   skel_img  = Skeletonized image
#   objects   = List of contours
#   mask      = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=skeleton, 
                                                 objects=edge_objects,
                                                 mask=cropped_mask)


# In[41]:


# Identify segments     

# Inputs:
#   skel_img  = Skeletonized image
#   objects   = List of contours
#   mask      = (Optional) binary mask for debugging. If provided, 
#               debug image will be overlaid on the mask.
segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=skeleton,
                                                       objects=leaf_obj,
                                                       mask=cropped_mask)


# In[42]:


# Similar to line thickness, there are optional text size and text thickness parameters 
# that can be adjusted to better suit images or varying sizes.
pcv.params.text_size=.2 # (default text_size=.55)
pcv.params.text_thickness=1 # (defaul text_thickness=2) 

segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=skeleton,
                                                       objects=leaf_obj,
                                                       mask=cropped_mask)


# In[43]:


# Measure path lengths of segments     

# Inputs:
#   segmented_img = Segmented image to plot lengths on
#   objects       = List of contours
labeled_img  = pcv.morphology.segment_path_length(segmented_img=segmented_img, 
                                                  objects=leaf_obj)


# In[44]:


# Measure euclidean distance of segments      

# Inputs:
#   segmented_img = Segmented image to plot lengths on
#   objects       = List of contours
labeled_img = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, 
                                                      objects=leaf_obj)


# In[45]:


# Measure curvature of segments      

# Inputs:
#   segmented_img = Segmented image to plot curvature on
#   objects       = List of contours
labeled_img = pcv.morphology.segment_curvature(segmented_img=segmented_img, 
                                               objects=leaf_obj)


# In[46]:


# Measure the angle of segments      

# Inputs:
#   segmented_img = Segmented image to plot angles on
#   objects       = List of contours
labeled_img = pcv.morphology.segment_angle(segmented_img=segmented_img, 
                                           objects=leaf_obj)


# In[47]:


# Measure the tangent angles of segments      

# Inputs:
#   segmented_img = Segmented image to plot tangent angles on
#   objects       = List of contours
#   size          = Size of ends used to calculate "tangent" lines
labeled_img = pcv.morphology.segment_tangent_angle(segmented_img=segmented_img, 
                                                   objects=leaf_obj, size=4)


# In[48]:


# Write morphological data to results file

# The print_results function will take the measurements stored when running any (or all) of these functions, format, 
# and print an output text file for data analysis. The Outputs class stores data whenever any of the following functions
# are ran: analyze_bound_horizontal, analyze_bound_vertical, analyze_color, analyze_nir_intensity, analyze_object, 
# fluor_fvfm, report_size_marker_area, watershed. If no functions have been run, it will print an empty text file 
pcv.print_results(filename=args.result)


# In[49]:



#CLEARS OUTPUTS
pcv.outputs.clear()

