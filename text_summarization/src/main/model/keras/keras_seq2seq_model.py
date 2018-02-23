

import pickle, random
import numpy as np

from __future__ import print_function
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from seq2seq.models import Seq2Seq
from main.resources import Loading

#loading  data as a list of jsons
#TODO: look into saving and loading the file as tf.record (if using tensorflow)
#TODO: make sure saved values are INTEGERS and not STRINGS

processed_text = Loading.load_pickle('/path/to/processed/data.txt')
word_to_int = Loading.load_pickle('/path/to/word_to_ind.txt')
int_to_word = Loading.load_pickle('/path/to/ind_to_word.txt')
embed_matrix = Loading.load_pickle('/path/to/embed_matrix.txt')

unique_features = len(word_to_int)
keys = ['content','title']
content = np.array([v for d in processed_text for k,v in d.items() if k == keys[0]]); content = content.astype('int64')
title = np.array([v for d in processed_text for k,v in d.items() if k == keys[1]]); title = title.astype('int64')


#shaping the parameters -> 1-hot encoding. Entire dataset doesn't fit into memory
#NOTE: sparse cross entropy requires index as inputs

#INPUTS
content_time_stamp = content.shape[1]
title_time_stamp = title.shape[1]
batch_size = 5
embedding=300
embedding_dim=300
weight_decay = .1
regularizer = l2(weight_decay) if weight_decay else None
lstm_layers = 11
lstm_size = 512
seed=911

#seed weight initialization
random.seed(seed)
np.random.seed(seed)

#create generator to feed in batches
def batch(content, title, batch_size):
    for i in range(0, len(content)-batch_size+1, batch_size):
        yield content[i:i+batch_size,:], title[i:i+batch_size,:]

def get_prediction(row):
    return ' '.join([int_to_word[ind] for ind in row])


model = Seq2Seq(batch_input_shape=(batch_size, content_time_stamp, unique_features), hidden_dim=lstm_size,
                output_length=title_time_stamp, output_dim=unique_features, depth=4, dropout=0.15)
optimizer = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()



#shaping the parameters -> 1-hot encoding. Entire dataset doesn't fit into memory
#NOTE: sparse cross entropy requires index as inputs

ident_matrix = np.eye(unique_features, dtype='bool')
ident_matrix_expand = np.expand_dims(ident_matrix,0)
identity_matrix = np.tile(ident_matrix_expand, (batch_size, 1, 1)) #(batch size, time_input, features)



training_loss_list = []
validation_loss_list = []
predicted_loss_list = []

for train_index, test_index in kfold.split(X_train):

    #spliting the training data into k-folds
    X_train_fold, X_validation = content[train_index], content[test_index]
    y_train_fold, y_validation = title[train_index], title[test_index]

    for content_batch, title_batch in batch(X_train_fold, y_train_fold, batch_size):

        # 1-hot encoding of inputs in batches (due to memory)
        # identity = (batch,unique_features,unique_ft), content = (batch, time, unique_ft),
        # title = (batch,timer,unique_ft)
        X = identity_matrix[:,content_batch[0],:]
        y = identity_matrix[:,title_batch[0], : ]  # (dim should be batch_size, 1-hot encoding)

        #define the checkpoint
        filepath="/home/paperspace/Documents/development/models/lstm/checkpoints/weights-improvement-TEST-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        #fit the model
        loss = model.train_on_batch(X,y)
        training_loss_list.append(loss)

        #save the model
        model.save_weights("/home/paperspace/Documents/development/models/lstm/model_weights_TEST.hdf5")

    #testing validation data
    number_samples_validation = X_validation.shape[0]
    for content_validation, title_validation in batch(X_validation, y_validation, number_samples_validation):

        XX = identity_matrix[:,content_validation[0],:]
        prediction = model.predict_on_batch(XX)
        prediction_argmax = np.argmax(prediction, axis=2)
        predicted_words = list(np.apply_along_axis(get_prediction, axis=1, arr=prediction_argmax))[0]

        yy = identity_matrix[:,title_validation[0], : ]  # (dim should be batch_size, 1-hot encoding)
        yy_argmax = np.argmax(yy, axis=2)[0]
        actual_words = get_prediction(yy_argmax)

        prediction_loss = model.test_on_batch(XX,yy)
        predicted_loss_list.append(prediction_loss)

        print('Predicted Title: {}'.format(predicted_words))
        print('Actual Title: {}'.format(actual_words))
        print('Prediction Loss: {}'.format(prediction_loss))

with open('/Documents/development/models/lstm/training_loss_list_TEST2.txt', 'wb') as f:
    pickle.dump(training_loss_list, f)

with open('/Documents/development/models/lstm/validation_loss_list_TEST2.txt', 'wb') as f:
    pickle.dump(validation_loss_list, f)

