# This is where you implement the U-Net model, using the tf.keras.Model package.
import tensorflow as tf

def extraction(filters, inputs): # first half of U, processing image and passing data to construction
	x = tf.Conv2D(filters, kernel_size(3,3), padding = 'same', strides = 1, activation= 'relu')(inputs)
	skip = tf.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation= 'relu')(x)
	pool = tf.MaxPooling2D(pool_size = (2,2), padding = 'same')(skip)
	return skip, pool

def base(filters, inputs): # bottom of U, processing data without skips
	x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
	x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
	return x
	
def construction(filters, skips, inputs): # recieves data from skips and processed data
	x = Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(inputs)
	skips = concatenate([x, skips], axis = -1)
	x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(skips)
	x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(x)
	return x