# This is where you implement the U-Net model, using the tf.keras.Model package.
import tensorflow as tf
import tensorflow.keras.layers as layers

def downconv(filters, inputs, last): # first half of U, processing image and passing data to construction
	x = layers.Conv2D(filters, kernel_size=(3,3), padding = 'same', strides = 1, activation= 'relu')(inputs)
	skip = layers.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation= 'relu')(x)
	if not last:
		x = layers.MaxPooling2D(pool_size = (2,2), padding = 'same')(skip)
		
	return skip, x

def base(filters, inputs): # bottom of U, processing data without skips
	x = layers.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
	x = layers.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
	return x
	
def upconv(filters, skips, inputs, last): # recieves data from skips and processed data
	x = layers.Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(inputs)
	crop_skip = layers.CenterCrop(height = x.shape[1], width = x.shape[2])(skips)
	skips = layers.Concatenate(axis=-1)([crop_skip, x])
	
	x = layers.Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(skips)
	x = layers.Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(x)
	
	return x

def unet():
    inputs = tf.keras.Input(shape=(640, 640, 1))
    
    #defining the encoder
    skip1, data1 = downconv(64, inputs, False)
    skip2, data2 = downconv(128, data1, False)
    skip3, data3 = downconv(256, data2, False)
    skip4, data4 = downconv(512, data3, True)
    
    #Setting up the baseline
    baseline = base(1024, data4)
    
    #Defining the entire decoder
    data1 = upconv(512, skip4, baseline, False)
    data2 = upconv(256, skip3, data1, False)
    data3 = upconv(128, skip2, data2, False)
    data4 = upconv(64, skip1, data3, True)
    
    #Setting up the output function for binary classification of pixels
    outputs = layers.Conv2D(1, 1, activation = 'sigmoid')(data4)
    
    #Finalizing the model
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'Unet')
    
    return model