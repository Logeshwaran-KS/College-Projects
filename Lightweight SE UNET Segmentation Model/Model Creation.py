IMG_DIM = (128,128,1)

#Squeeze and Excitation Block
def squeeze_excite_block(input_tensor, ratio=16):
    squeeze = GlobalAveragePooling2D()(input_tensor)

    excitation = Dense(units=int(input_tensor.shape[-1] / ratio), activation='relu')(squeeze)
    excitation = Dense(units=input_tensor.shape[-1], activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, input_tensor.shape[-1]))(excitation)

    scaled_input = Multiply()([input_tensor, excitation])

    return scaled_input

# convolution Layer
def conv2d_block( input_tensor, n_filters, kernel_size = (3,3), name="contraction"):
  "Add 2 conv layer"
  x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal',
             padding='same',activation="relu", name=name+'_1')(input_tensor)
  x = squeeze_excite_block(x)
  return x


inp = Input( shape=IMG_DIM )

d = conv2d_block( inp, 32, name="contraction")
p = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d)
p = BatchNormalization(momentum=0.8)(p)
p = Dropout(0.2)(p)

d1 = conv2d_block( p, 64, name="contraction_1")
p1 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d1)
p1 = BatchNormalization(momentum=0.8)(p1)
p1 = Dropout(0.2)(p1)

d2 = conv2d_block( p1, 128, name="contraction_2_1" )
p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d2)
p2 = BatchNormalization(momentum=0.8)(p2)
p2 = Dropout(0.2)(p2)

d3 = conv2d_block( p2, 256, name="contraction_3_1")
p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d3)
p3 = BatchNormalization(momentum=0.8)(p3)
p3 = Dropout(0.2)(p3)

d4 = conv2d_block(p3,512, name="contraction_4_1")
p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d4)
p4 = BatchNormalization(momentum=0.8)(p4)
p4 = Dropout(0.2)(p4)

d5 = conv2d_block(p4,512, name="contraction_5_1")

u1 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(d5)
u1 = concatenate([u1,d4])
u1 = Dropout(0.2)(u1)
c1 = conv2d_block(u1, 512, name="expansion_1")

u2 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(c1)
u2 = concatenate([u2,d3])
u2 = Dropout(0.2)(u2)
c2 = conv2d_block(u2, 256, name="expansion_2")

u3 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(c2)
u3 = concatenate([u3,d2])
u3 = Dropout(0.2)(u3)
c3 = conv2d_block(u3, 128, name="expansion_3")

u4 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(c3)
u4 = concatenate([u4,d1])
u4 = Dropout(0.2)(u4)
c4 = conv2d_block(u4,64, name="expansion_4")

u5 = Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same')(c4)
u5 = concatenate([u5,d])
u5 = Dropout(0.2)(u5)
c5 = conv2d_block(u5,32, name="expansion_5")

out = Conv2D(1, (1,1), name="output", activation='sigmoid')(c5)

unet = Model( inp, out )
unet.summary()