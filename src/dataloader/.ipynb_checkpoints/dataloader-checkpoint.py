# This is where you write functions that mask the images based on the COCO format annotations. This is also where you use the tf.data package to
# build your input pipeline that converts the jpg images to TensorFlow tensors (tf.Tensor).
import tensorflow as tf
from criterion import *
import pathlib

def load_image(image, mask): # prepares data for use in model
  input_image = tf.image.resize(image, (128, 128))
  input_mask = tf.image.resize(mask, (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
  )

  return input_image, input_mask


def build_data(data): # creates the image and mask tensors for the training and validation sets
	train_dir = pathlib.Path(r"C:/Users/tyler/content/brain_tumor_dataset/brain-tumor-image-dataset-semantic-segmentation/versions/1/train/")
	val_dir = pathlib.Path(r"C:/Users/tyler/content/brain_tumor_dataset/brain-tumor-image-dataset-semantic-segmentation/versions/1/valid/")
		
	coco = COCO(r"C:\Users\tyler\content\brain_tumor_dataset\brain-tumor-image-dataset-semantic-segmentation\versions\1\train\_annotations.coco.json")

	train_imgs = list(train_dir.glob('**/*.jpg')) # collections of paths for creating masks
	val_imgs = list(val_dir.glob('**/*.jpg'))

	train_ds = tf.data.Dataset.list_files(str(train_dir)+"/*.jpg", shuffle=False) # creating image tensors
	val_ds = tf.data.Dataset.list_files(str(val_dir)+"/*.jpg", shuffle=False) # maintains same order as path collection
	
	training_masks = build_masks(train_imgs, coco, data)
	val_masks = build_masks(val_imgs, coco, data)

	
	def build_masks(paths, coco, data): # creates masks for the given image paths in order
	    mask_ds = []
		
		print("=== creating masks === ") # ~ 1 minute
		
	    for i in range(len(paths)): # creates masks for every path
	
	        ID = get_id_from_path(data, str(paths[i])) # in criterion.py
			
	        cat_ids = coco.getCatIds()
	        anns_ids = coco.getAnnIds(imgIds=ID, catIds=cat_ids, iscrowd=None) # ID
	        anns = coco.loadAnns(anns_ids)
	
	        mask = coco.annToMask(anns[0])
	        for j in range(len(anns)): # in case there are more than one annotation, yet to encounter any
	            mask += coco.annToMask(anns[j])
	
	        mask_ds.append(mask)
	
		print("=== creating tensor ===") # ~ 4 minutes
	    return tf.data.Dataset.from_tensor_slices(mask_ds)

	
	
