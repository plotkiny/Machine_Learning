

import sys, pickle, random
import numpy as np

from __future__ import print_function
from keras.layers import Activation, Dense, Dropout, Input, LSTM
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#loading  data as a list of jsons
#TODO: look into saving and loading the file as tf.record (if using tensorflow)
#TODO: make sure saved values are INTEGERS and not STRINGS

with open("/home/paperspace/Documents/code/pipeline3/processed_data-20K_vocab-75th_percentile-with_sampling.txt", "rb") as f:   # Unpickling
    processed_text = pickle.load(f)

with open("/home/paperspace/Documents/data/word_to_ind.txt", "rb") as f:   # Unpickling
    word_to_int = pickle.load(f)
    
with open("/home/paperspace/Documents/data/ind_to_word.txt", "rb") as f:   # Unpickling
    int_to_word = pickle.load(f)
    
unique_features = len(word_to_int)
keys = ['content','title']    
content = np.array([v for d in processed_text for k,v in d.items() if k == keys[0]]); content = content.astype('int64')
title = np.array([v for d in processed_text for k,v in d.items() if k == keys[1]]); title = title.astype('int64')


#INPUTS
#shaping the parameters -> 1-hot encoding. Entire dataset doesn't fit into memory
#NOTE: sparse cross entropy requires index as inputs

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


#shaping the parameters -> 1-hot encoding. Entire dataset doesn't fit into memory
#NOTE: sparse cross entropy requires index as inputs

content_time_stamp = content.shape[1]
title_time_stamp = title.shape[1]

ident_matrix = np.eye(unique_features, dtype='bool')
ident_matrix_expand = np.expand_dims(ident_matrix,0)
identity_matrix = np.tile(ident_matrix_expand, (batch_size, 1, 1)) #(batch size, time_input, features)

#create generator to feed in batches
def batch(content, title, batch_size):
    for i in range(0, len(content)-batch_size+1, batch_size):
        yield content[i:i+batch_size,:], title[i:i+batch_size,:]
        
print('Build model...')

#TODO: adding COnv1d to input and output laters
main_input = Input(shape=(content_time_stamp, unique_features), dtype='float32', name='main_input')
encoder = LSTM(lstm_size, input_shape=(content_time_stamp, unique_features), return_sequences=False)(main_input)  #output = (batch_size, lstm_size)
encoder = RepeatVector(title_time_stamp)(encoder)
for _ in range(title_time_stamp):
    encoder = LSTM(lstm_size, return_sequences=True)(encoder)
    encoder = Dropout(.15)(encoder)
decoder = LSTM(lstm_size, return_sequences=True)(encoder)
main_output = TimeDistributed(Dense(unique_features, activation='softmax', name='main_output'))(encoder)
model = Model(inputs=main_input, outputs=main_output)

model.summary()
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# 1-hot encoding of inputs in batches (due to memory)
for content, title in batch(content, title, batch_size):
    
    # i = 1 x 40k x 40k, c = (1, 160, 40004), t = (1, 15, 40004)
    X = identity_matrix[:,content[0],:]
    y = identity_matrix[:,title[0], : ]  # (dim should be batch_size, 1-hot encoding)

    #fit the model
    model.fit(X, y, batch_size=1, epochs=15)
    
    #define the checkpoint
    filepath="/home/paperspace/Documents/code/pipeline1/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    #Picking random sequence from the batch and testing our predictions
    #sampling from the batch a set number of times
    for test in range(batch_size):
        rndm_index = random.randint(0,batch_size-1)  #get random sample from the batch
        rndm_sample = content[rndm_index,]
     
        identity_sample = identity_matrix[0,rndm_sample.astype('int64')]
        identity_sample_extend = np.expand_dims(identity_sample,0)
        XX = np.tile(identity_sample_extend, (1, 1, 1)) #(batch size, time_input, features)
        prediction = model.predict(XX, verbose=0)






        
     