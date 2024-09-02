# Importing Data
images=np.load('images.npy')
masks=np.load('masks.npy')

#Data Split
x_train, x_test, y_train, y_test= train_test_split (images,masks,test_size=0.2, shuffle= True)
x_train.shape

#Data Augument
x_train= np.append( x_train, [ np.fliplr(x) for x in  x_train], axis=0 )
y_train = np.append( y_train, [ np.fliplr(y) for y in  y_train], axis=0 )

#Data Generator
train_datagen = ImageDataGenerator(brightness_range=(0.9,1.1),
                                   zoom_range=[.9,1.1],
                                   fill_mode='nearest')
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
val_generator = val_datagen.flow(x_test, y_test, batch_size=32)