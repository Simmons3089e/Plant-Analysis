#!/usr/bin/env python
# coding: utf-8

# # Morphology Tutorial
# Morphology sub-package functions can be used with a clean mask of a plant (see VIS tutorial for examples of masking background. This tutorial will start with a binary mask (after object segmentation has been completed) but in a complete workflow users will need to use other functions to achieve plant isolation. Skeletonizing is very sensitive to any pepper noise remaining within a binary mask. Morphology functions are intended to be one type of object analysis. These functions can potentially return information about leaf length, leaf angle, and leaf curvature.

# In[104]:


# Import libraries
from plantcv import plantcv as pcv 
import numpy as np
import cv2


# In[105]:


class options:
    def __init__(self):
        self.image = "C:/Users/Beth/Downloads/14-24-00.jpg"
        self.debug = "plot"
        self.writeimg= False 
        self.result = "./morphology_tutorial_results2.json"
        self.outdir = "."

# Get options
args = options()

# Set debug to the global parameter 
pcv.params.debug = args.debug


# In[106]:


# Read image (sometimes you need to run this line twice to see the image) 

# Inputs:
#   filename - Image file to be read in 
#   mode - How to read in the image; either 'native' (default), 'rgb', 'gray', or 'csv'
img, path, filename = pcv.readimage(filename=args.image)


# In[107]:


#crop Image

from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)

pcv.params.debug = "plot"

# Crop image
crop_img = pcv.crop(img=img, x=400, y=20, h=100, w=100)


# In[108]:


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


# In[109]:


# Convert RGB to HSV and extract the saturation channel
# Then set threshold for saturation
s = pcv.rgb2gray_hsv(rgb_img=img1, channel='s')
s_thresh = pcv.threshold.binary(gray_img=s, threshold=122, max_value=255, object_type='dark')


# In[110]:


# Set Median Blur
#Input box size "ksize" 
s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=2)
s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=2)


# In[111]:


# Convert RGB to LAB and extract the blue channel
#Then threshold the image
b = pcv.rgb2gray_lab(rgb_img=img1, channel='b')
b_thresh = pcv.threshold.binary(gray_img=b, threshold=135, max_value=255, object_type='light')

#Setting threshold continued
b_cnt = pcv.threshold.binary(gray_img=b, threshold=135, max_value=255, object_type='light')


# In[112]:


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
roi1, roi_hierarchy= pcv.roi.rectangle(img=img1, x=75, y=60, h=20, w=20)


# In[113]:


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


# In[114]:


obj, mask = pcv.object_composition(img=crop_img, contours=roi_objects, hierarchy=hierarchy)

######## workflow steps here ########

# Find shape properties, output shape image (optional)
shape_img = pcv.analyze_object(crop_img, leaf_obj, cropped_mask)

#pcv.outputs.add_observation(variable='Plant Solidity', trait='Solidity',
#                           method='none', scale='percent', datatype=float,
#value=shape_img, label='percent')

# Look at object area data without writing to a file 
#plant_area = pcv.outputs.observations['Pixels']['value']


# In[115]:


#Area of plant
#Fill in segments (also stores out area data)  

# Inputs:
# mask         = Binary image, single channel, object = 1 and background = 0
# objects      = List of contours
filled_img = fill_segments(mask=cropped_mask, objects=roi_objects)

# Access data stored out from fill_segments
segments_area = pcv.outputs.observations['segment_area']['value']


# In[116]:


#Length
# Measure path lengths of segments     

# Inputs:
#   segmented_img = Segmented image to plot lengths on
#   objects       = List of contours
labeled_img  = pcv.morphology.segment_path_length(segmented_img=filled_img, 
                                                  objects=roi_objects)


# In[117]:


# Write morphological data to results file

# The print_results function will take the measurements stored when running any (or all) of these functions, format, 
# and print an output text file for data analysis. The Outputs class stores data whenever any of the following functions
# are ran: analyze_bound_horizontal, analyze_bound_vertical, analyze_color, analyze_nir_intensity, analyze_object, 
# fluor_fvfm, report_size_marker_area, watershed. If no functions have been run, it will print an empty text file 
pcv.print_results(filename=args.result)


# In[100]:


#CLEARS OUTPUTS
#pcv.outputs.clear()

