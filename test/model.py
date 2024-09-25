import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     Conv2DTranspose, UpSampling2D, Reshape, Input, concatenate)

def spi_nn():
    input_img = Input(shape=(128, 128, 1))

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Flatten and encode
    x = Flatten()(x)
    encoded = Dense(128, activation='relu')(x)

    # Decoder
    x = Dense(16 * 16 * 256, activation='relu')(encoded)
    x = Reshape((16, 16, 256))(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Output layer with tanh activation for normalized output
    decoded = Conv2DTranspose(1, (3, 3), activation='tanh', padding='same')(x)

    encdec = models.Model(input_img, decoded)
    return encdec

def encoding_block(layer, n_filters, use_dropout=False):
    """
    Encoder block for UNet 
    """
    conv = Conv2D(n_filters, 3, activation='relu', padding='same')(layer)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same')(conv)
    
    if use_dropout:
        conv = Dropout(0.5)(conv)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return pool, conv

def decoding_block(layer, skip_layer, n_filters):
    """
    Decoder block for UNet 
    """
    upsampled = UpSampling2D(size=(2, 2))(layer)
    conv_upsampled = Conv2D(n_filters, 2, activation='relu', padding='same')(upsampled)
    concat = concatenate([skip_layer, conv_upsampled], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same')(concat)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same')(conv)
    
    return conv

# UNet Model - Denoising&Demasking
def unet():
    inputs = Input(shape=(128, 128, 1))
    
    # Encoding (Downsampling)
    layer1, skip_layer1 = encoding_block(inputs, 64)
    layer2, skip_layer2 = encoding_block(layer1, 128)
    layer3, skip_layer3 = encoding_block(layer2, 256)
    layer4, skip_layer4 = encoding_block(layer3, 512, use_dropout=True)

    # Bottleneck
    bottleneck = Conv2D(1024, 3, activation='relu', padding='same')(layer4)
    bottleneck = Conv2D(1024, 3, activation='relu', padding='same')(bottleneck)
    bottleneck = Dropout(0.5)(bottleneck)

    # Decoding (Upsampling)
    layer6 = decoding_block(bottleneck, skip_layer4, 512)
    layer7 = decoding_block(layer6, skip_layer3, 256)
    layer8 = decoding_block(layer7, skip_layer2, 128)
    layer9 = decoding_block(layer8, skip_layer1, 64)

    # Output layer with tanh activation
    output = Conv2D(2, 3, activation='relu', padding='same')(layer9)
    output = Conv2D(1, 1, activation='tanh')(output)
    
    # UNet Model
    model = models.Model(inputs, output)
    return model
