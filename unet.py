import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, \
                         LeakyReLU, Conv3DTranspose, Multiply, Lambda, Add, Subtract, Dense, Reshape, concatenate, Dropout
from fmLayer import CalFMLayer
from NDI import NDIErr
import tensorflow.keras.backend as K
 
K.set_image_data_format("channels_first")



def semi_model_t(unet):
    fm_in1 = Input((1, None, None, None))
    mask1  = Input((1, None, None, None))
    qsm_kernel = Input((1, None, None, None))
    w1 = Input((1, None, None, None))
    
    fm_in2 = Input((1, None, None, None))
    m2 = Input((1, None, None, None))
    ma = Input((1, None, None, None))

    suscp_out1 = unet([fm_in1, mask1])
    suscp_out2 = unet([fm_in2, mask1])

    fm1 = CalFMLayer()([suscp_out1, qsm_kernel]) # do dipole convolution
    fm1 = Multiply()([fm1, mask1])
    err_fm1 = NDIErr()([fm1, fm_in1, w1])
    
    diff_suscp = Subtract()([suscp_out2, suscp_out1])
    diff_suscp_roi = Multiply()([diff_suscp, m2])
    diff_suscp_out = Multiply()([diff_suscp, ma])

    model = Model(inputs=[fm_in1, mask1, w1, qsm_kernel, fm_in2, m2, ma],
                  outputs=[err_fm1, suscp_out1, diff_suscp_roi, diff_suscp_out])
    return model


def unet_model_3d(pool_size=(2, 2, 2), 
                  n_outputs=1, 
                  deconvolution=False,
                  kernel=(3,3,3),
                  depth=5, 
                  n_base_filters=32, 
                  batch_normalization=True, 
                  activation_name="linear"):

    fm_in = Input((1, None, None, None))
    mask = Input((1, None, None, None))
    
    current_layer = concatenate([fm_in, mask], axis=1) 

    levels = list()

    for layer_depth in range(depth):
        n_filters=n_base_filters*(2**layer_depth)
        layer = create_convolution_block(input_layer=current_layer,
                                          kernel = kernel,
                                          n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization,
                                          dilation_rate=(1,1,1))
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer)
            levels.append([layer, current_layer])
        else:
            current_layer = layer
            levels.append([layer])
   

    for layer_depth in range(depth-2, -1, -1):
        n_filters = n_filters//2
        up_convolution = get_up_convolution(pool_size=pool_size,
                                            kernel_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=n_filters)(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][0]], axis=1)

        current_layer = create_convolution_block(n_filters=n_filters,
                                                 kernel=kernel,
                                                 input_layer=concat,
                                                 batch_normalization=batch_normalization,
                                                 dilation_rate=(1,1,1))
         

    op = Conv3D(n_outputs, [1,1,1], padding='same')
    out = op(current_layer)
    out = Activation(activation_name)(out)
    out = Multiply()([out, mask])


    model = Model(inputs=[fm_in, mask],
                  outputs=[out])

    return  model

def create_convolution_block(input_layer, n_filters, batch_normalization=True, 
                             kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), dilation_rate=(1,1,1)):
    op = Conv3D(n_filters, kernel, padding='same', strides=strides, dilation_rate=dilation_rate)
    
    layer = op(input_layer)
    
    if batch_normalization:
        layer = BatchNormalization(axis=1, momentum=0.99)(layer)
    
    return LeakyReLU(0.1)(layer)

def get_up_convolution(n_filters, pool_size, kernel_size=(2,2,2), strides=(2, 2, 2), deconvolution=True):
    if deconvolution:
        return Conv3DTranspose(filters=n_filters, 
                               padding = 'same',
                               kernel_size=kernel_size,
                               strides=strides,
                               use_bias=False)
    else:
        return UpSampling3D(size=pool_size)
