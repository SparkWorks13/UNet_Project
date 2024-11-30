# This is where you write test functions to verify the functionality of your code. This does not necessarily need to happen all in 
# one file

from dataloader import *
from criterion import *
import json
from pycocotools.coco import COCO
import pytest
import numpy as np

def test_input_pipeline():
	# initializing annotation data for training and validation datasets
	with open(r"C:\Users\tyler\content\brain_tumor_dataset\brain-tumor-image-dataset-semantic-segmentation\versions\1\train\_annotations.coco.json", 'r') as file:
	    data_t = json.load(file)

	with open(r"C:\Users\tyler\content\brain_tumor_dataset\brain-tumor-image-dataset-semantic-segmentation\versions\1\valid\_annotations.coco.json", 'r') as file:
	    data_v = json.load(file)

	# initializing annotations for both datasets
	coco_t = COCO(r"C:\Users\tyler\content\brain_tumor_dataset\brain-tumor-image-dataset-semantic-segmentation\versions\1\train\_annotations.coco.json")
	coco_v = COCO(r"C:\Users\tyler\content\brain_tumor_dataset\brain-tumor-image-dataset-semantic-segmentation\versions\1\valid\_annotations.coco.json")
	
	t_data, v_data = build_data(data_t, data_v) # converting data to tensors
	
	t_data['data'] = list(t_data['data'].as_numpy_iterator()) # converting tensors to arrays for analysis
	t_data['masks'] = list(t_data['masks'].as_numpy_iterator())
	v_data['data'] = list(v_data['data'].as_numpy_iterator())
	v_data['masks'] = list(v_data['masks'].as_numpy_iterator())
	
	index = 12 # arbitrary
	
	ID_t = get_id_from_path(data_t, t_data['data'][12].decode('utf-8')) # retrieving ID corresponding to selected index
	ID_v = get_id_from_path(data_v, v_data['data'][12].decode('utf-8'))
	
	path_t = data_t['images'][ID_t]['file_name'] # retrieving path from determined ID
	path_v = data_v['images'][ID_v]['file_name']

	mask_t = create_mask(ID_t, coco_t) # created masks from determined ID
	mask_v = create_mask(ID_v, coco_v)
	
	assert path_t in t_data['data'][12].decode('utf-8') and path_v in v_data['data'][12].decode('utf-8') # testing correct image for ID
	assert np.equal(t_data['masks'][12], mask_t).all() and np.equal(v_data['masks'][12], mask_v).all() # testing correct mask for ID

	print("PASS")
