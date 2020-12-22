
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils import plot_model

#load the dataset(X_data), the label (y_data), and the test data (X_test) 
X_data = pd.read_csv("X_data.txt", header=None, names=["text"])
y_data = pd.read_csv("y_data.txt", header=None, names=["label"])
X_test = pd.read_csv("X_test.txt", header=None, names=["text"])

#extract the values
X = X_data["text"].values
y = y_data["label"].values
X_t = X_test["text"].values

#split the dataset to training, validation set 80:20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=666, stratify=y)

#initialize the vectorizer and fit the training set
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(X_train)

#transform all the data into vector
X_train = vectorizer.transform(X_train)
X_val  = vectorizer.transform(X_val)
X_t = vectorizer.transform(X_t)

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(Dense(16, input_dim=input_dim, activation='relu', 
                kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), 
                kernel_initializer='glorot_normal'))
model.add(Dense(1, activation='sigmoid', 
                kernel_regularizer=regularizers.l2(0.01), 
                kernel_initializer='glorot_uniform'))
optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
model.summary()

dot_img_file = 'model.png'
plot_model(model, to_file=dot_img_file, show_shapes=True)

history = model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val), 
                    batch_size=10)

pred_model = (model.predict(X_val) > 0.5).astype("int32")
print("Accuracy %s" % accuracy_score(pred_model,y_val))
print(classification_report(y_val,pred_model))


plt.style.use('ggplot')

#function to plot the accuracy and loss per epoch
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("plot.png")
    
plot_history(history)

# Run prediction of the test set with the model
digit = ((model.predict(X_t) > 0.5).astype("int32"))

#function to write the prediction into anwer.txt file
def save_pred(pred):
    fp = open("answer.txt", "w", newline='\n')
    for num in pred:
        text = (str(num)).strip("[ ]") + "\n"
        fp.write(text)
    fp.close()
    
save_pred(digit)
