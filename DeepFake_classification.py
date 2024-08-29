import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from keras.preprocessing import image
from keras.layers import Dropout


data_train_path = 'Deepfake/Train'
data_test_path = 'Deepfake/Test'
data_val_path = 'Deepfake/Validation'


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)



train_flow = train_datagen.flow_from_directory(data_train_path, target_size=(64, 64), batch_size=32, class_mode='categorical')
validation_flow = validation_datagen.flow_from_directory(data_val_path, target_size=(64, 64), batch_size=32, class_mode='categorical')
test_flow = test_datagen.flow_from_directory(data_test_path, target_size=(64, 64), batch_size=32, class_mode='categorical')




model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(2, activation='softmax'))




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_flow, epochs=10, validation_data=validation_flow)


loss, accuracy = model.evaluate(test_flow)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


loss, accuracy = model.evaluate(train_flow)
print(f'Train Accuracy: {accuracy * 100:.2f}%')


test_image = image.load_img('fake_img.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
train_flow.class_indices


result = model.predict(test_image)
predicted_class = np.argmax(result[0])


if predicted_class == 0:
    prediction = 'Fake'
else:
    prediction = 'Real'


print(f"The predicted class is: {prediction}")
