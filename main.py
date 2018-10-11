from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import ZeroPadding2D, Activation
from keras.metrics import top_k_categorical_accuracy
import matplotlib.pyplot as plt
import numpy as np
from os import path
from glob import glob

from keras.preprocessing.image import load_img, img_to_array

def plot_img(img, img_index, label_name):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.set_size_inches(0.32, 0.32) # fig.set_size_inches(1, 1)
    plt.imshow(img, cmap='binary')
    plt.axis('off')

    # plt.show()

    # save
    # https://stackoverflow.com/a/31954057
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    plt.savefig('test_set/{}-{}.png'.format(img_index, label_name))

def show_train_history(train_history):
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('1.png')
    plt.clf()

    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('2.png')
    plt.clf()
    
    plt.plot(train_history.history["top_5_accuracy"])
    plt.plot(train_history.history["val_top_5_accuracy"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Top 5 Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('3.png')
    plt.clf()

def top_5_accuracy(y_true, y_pred):
    # https://keras.io/metrics/#top_k_categorical_accuracy
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def training():
    print()

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32,32,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax')) # softmax: [0, 1]
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_5_accuracy])
    train_history = model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.2, epochs=50, batch_size=32, verbose=2)
    show_train_history(train_history)
    model.save('model.h5')

def testing():
    model = load_model('model.h5', custom_objects={'top_5_accuracy': top_5_accuracy})

    ## Accuracy ##
    scores = model.evaluate(x_test_normalize, y_test_onehot)
    print()
    # print("Accuracy=", scores)
    top_1_error_rate = 1-scores[1]
    print("top 1 error rate = {:.3f} ({:.2f}%)".format(top_1_error_rate, top_1_error_rate*100))
    top_5_error_rate = 1-scores[2]
    print("top 5 error rate = {:.3f} ({:.2f}%)".format(top_5_error_rate, top_5_error_rate*100))
    print()

    ## Predict images ##
    # base_dir = path.abspath(path.dirname(__file__))
    # relative_path = "test_set_100".format()
    # regex_full_path = path.join(base_dir, relative_path + "/*")
    # file_names = glob(regex_full_path)
    # imgs = []
    # for file_name in file_names:
    #     img = load_img(file_name, target_size=(32, 32))
    #     x = img_to_array(img)
    #     x_normalize = x / 255.0
    #     imgs.append(x_normalize)
    # print("## Top 5")
    # all_predictions = model.predict(np.array(imgs))
    # for i, all_prediction in enumerate(all_predictions):
    #     top_values_index = sorted(range(len(all_prediction)), key=lambda i: all_prediction[i])[-5:] #https://stackoverflow.com/a/49827994
    #     top_values_index = reversed(top_values_index)
    #     print('file: {:20}| '.format(path.basename(file_names[i])), end='')
    #     for label in top_values_index:
    #         print('Label: {}({}). '.format(label, label_dict[label]), end='')
    #     print()
    # print("## Top 1")
    # all_prediction = model.predict_classes(np.array(imgs))
    # for i, label in enumerate(all_prediction):
    #     print('file: {:20}| Label: {}({}).'.format(path.basename(file_names[i]), label, label_dict[label]))

    ## Predict single image ##
    # imgs = []
    # img = load_img('test_set_100/1-truck.png', target_size=(32, 32))
    # x = img_to_array(img)
    # x_normalize = x / 255.0
    # imgs.append(x_normalize)
    # print("## Top 5")
    # all_prediction = model.predict(np.array(imgs))
    # all_prediction = all_prediction[0]
    # top_values_index = sorted(range(len(all_prediction)), key=lambda i: all_prediction[i])[-5:] #https://stackoverflow.com/a/49827994
    # for i in reversed(top_values_index):
    #     print('Label: {}, {}. '.format(i, label_dict[i]), end='')
    # print()
    # print("## Top 1")
    # all_prediction = model.predict_classes(np.array(imgs))
    # print('Label: {}, {}.'.format(all_prediction[0], label_dict[all_prediction[0]]))

def main():
    training()
    testing()
    pass

## Global Variables ##

label_dict = {
    0: "airplain", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9:"truck"
}

# ~/.keras/datasets/cifar-10-batches-py/
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(" x_train.shape, y_train.shape, x_test.shape, y_test.shape\n", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# img_index = 1
# print(y_train[img_index][0], label_dict[y_train[img_index][0]])
# print(x_train[img_index][0]) # first row of second image
# plot_img(x_train[img_index], img_index, label_dict[y_train[img_index][0]]) # plot second image

# save test_set/
# for i in range(0, 10):
#     plot_img(x_train[i], i, label_dict[y_train[i][0]]) # plot second image

# Normalization
# e.g. 255/255.0=1, 51/255.0=0.2
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

# e.g. [9] => [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
# print(y_train_onehot[1])

if __name__ == '__main__':
    main()