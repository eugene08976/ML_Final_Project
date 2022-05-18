"""
Description: Train emotion classification model
"""
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import big_XCEPTION
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam, Nadam
import matplotlib.pyplot as plt
import os

# parameters
batch_size = 64
num_epochs = 1000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'checkpoints/'

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

# callbacks
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1,
                                  patience=int(patience/4), verbose=1)
model_names = base_path + 'weight.{epoch:02d}.{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_acc', 
                          verbose=1, save_best_only=True)
callbacks = [model_checkpoint, early_stop, reduce_lr]

# model parameters/compilation
model = big_XCEPTION(input_shape, num_classes)
# model = vgg16(input_shape, num_classes, lr=0.01)
optimizer = Nadam(learning_rate= 0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['acc'])



def checkpoint_function():
    if not os.listdir(base_path):
        return
        
    files_int = list()
    for i in os.listdir(base_path):
        epoch = int(i.split('.')[1])
        files_int.append(epoch)
   
    max_value = max(files_int)
    for i in os.listdir(base_path):
        epoch = int(i.split('.')[1])
        if epoch > max_value:
            pass
        elif epoch < max_value:
            pass
        else:
            final_file = i    
       
    return final_file, max_value


checkpoint_path_file = checkpoint_function()







if checkpoint_path_file is not None:
     # Load model:
    checkpoint_path_file = checkpoint_function()[0]
    max_value = checkpoint_function()[1]

    model = load_model(os.path.join(base_path, checkpoint_path_file))
    hist = model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest), initial_epoch = max_value)
    hist = hist.history
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.title("Accuracy plot")
    plt.legend(["train","test"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("bigx_ndam_accuracy.png")

else:
    hist = model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
    hist = hist.history
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.title("Accuracy plot")
    plt.legend(["train","test"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("bigx_ndam_accuracy.png")
