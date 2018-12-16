from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, Activation

def denoise_model():
    image_shape=(None,None,1)
    act='relu'
    model=Sequential()
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=1,input_shape=image_shape))
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    # model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    # model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=4))
    # model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=3))
    # model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(64,(3,3),padding='same',dilation_rate=2))
    # model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Conv2D(1,(3,3),padding='same',dilation_rate=1))

    return model
