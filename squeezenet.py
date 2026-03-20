from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D
import keras.backend as K


def SqueezeNet_11(input_shape, nb_classes, dropout_rate=None, compression=1.0, kernel_initializer="glorot_uniform"):

    
    input_img = Input(shape=input_shape)
    
    # 64 means 64 filters
    x = Conv2D(int(64*compression), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1', kernel_initializer=kernel_initializer)(input_img)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_fire_module(x, int(16*compression), name='fire2', kernel_initializer=kernel_initializer)
    x = create_fire_module(x, int(16*compression), name='fire3', kernel_initializer=kernel_initializer)
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool3')(x)
    
    x = create_fire_module(x, int(32*compression), name='fire4', kernel_initializer=kernel_initializer)
    x = create_fire_module(x, int(32*compression), name='fire5', kernel_initializer=kernel_initializer)
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool5')(x)
    
    x = create_fire_module(x, int(48*compression), name='fire6', kernel_initializer=kernel_initializer)
    x = create_fire_module(x, int(48*compression), name='fire7', kernel_initializer=kernel_initializer)
    x = create_fire_module(x, int(64*compression), name='fire8', kernel_initializer=kernel_initializer)
    x = create_fire_module(x, int(64*compression), name='fire9', kernel_initializer=kernel_initializer)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    # Creating last conv10
    x = output(x, nb_classes, kernel_initializer=kernel_initializer)

    return Model(inputs=input_img, outputs=x)


def output(x, nb_classes, kernel_initializer="glorot_uniform"):
    x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D(name='avgpool10')(x)
    x = Activation("softmax", name='softmax')(x)
    return x


def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False, kernel_initializer="glorot_uniform"):

    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name, kernel_initializer=kernel_initializer)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name, kernel_initializer=kernel_initializer)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name, kernel_initializer=kernel_initializer)(squeeze)
    
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret


def get_axis():
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    return axis