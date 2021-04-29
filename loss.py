import tensorflow.keras.backend as K


def gradient(x):
    assert K.ndim(x) == 5
    if K.image_data_format() == 'channels_first':
        dx = (x[:, :, :-1, :-1, :-1] - x[:, :, 1:, :-1, :-1])
        dy = (x[:, :, :-1, :-1, :-1] - x[:, :, :-1, 1:, :-1])
        dz = (x[:, :, :-1, :-1, :-1] - x[:, :, :-1, :-1, 1:])
    else:
        dx = (x[:, :-1, :-1, :-1, :] - x[:, 1:, :-1, :-1, :])
        dy = (x[:, :-1, :-1, :-1, :] - x[:, :-1, 1:, :-1, :])
        dz = (x[:, :-1, :-1, :-1, :] - x[:, :-1, :-1, 1:, :])
    return dx, dy, dz

def tv_loss(y_true, y_pred):
    dx, dy, dz = gradient(y_pred)
    tv_loss = K.mean(K.abs(K.abs(dx))) + \
              K.mean(K.abs(K.abs(dy))) + \
              K.mean(K.abs(K.abs(dz)))
    return tv_loss