from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
#размер входящего изображения
img_width, img_height = 150, 150
train_data_dir = 'train'
test_data_dir = 'test'
valid_data_dir = 'validation'
#кол-во тренировочной выборки
nb_train_samples = 2520
#кол-во тестовой выборки
nb_test_samples = 372
#кол-во валидационной выборки
nb_valid_samples = 33
#разделение
batch_size = 126
#кол-во эпох
epochs = 10
#tensorflow channels:Last
#размер тензора
input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

val_generator=datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width,img_height),
    batch_size=10,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_valid_samples//10)

scores = model.evaluate_generator(test_generator,nb_test_samples//batch_size)
print('Accuracy test data: %.2f%%' % (scores[1]*100))