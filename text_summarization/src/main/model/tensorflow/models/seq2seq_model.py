#!usr/bin/env/python

import os, time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from main.resources.helper_function import Loading
from sklearn.model_selection import KFold
from tqdm import tqdm

class Seq2Seq(object):
    
    def __init__(self, configuration, embedding, v_to_i, i_to_v, mode):
        
        #additional required arguments
        self.embeddings = embedding
        self.vocab_to_int = v_to_i
        self.int_to_vocab = i_to_v
        self.mode = mode
                        
        assert self.mode in ['training', 'evaluation', 'inference']

        self.batch_size = configuration['batch_size']
        self.checkpoint = configuration['checkpoint_directory']
        self.display_step = configuration['display_step']
        self.epochs = configuration['epochs']
        self.keep_probability = configuration['keep_probability']
        self.starter_learning_rate = configuration['learning_rate']
        self.learning_rate_decay = configuration['learning_rate_decay']
        self.num_layers = configuration['num_layers']
        self.min_learning_rate = configuration['min_learning_rate']
        self.adaptive_optimizer = configuration['adaptive_optimizer']
        self.per_epoch = configuration['per_epoch']
        self.attn_size = configuration['attention_size']
        self.rnn_size = configuration['rnn_size']
        self.stop = configuration['stop']
        self.stop_early = configuration['stop_early']
        self.fold = configuration['fold']
        self.vocab_size = len(self.vocab_to_int)+1
        self.beam_search = True
        self.beam_length = configuration['beam_length']
        self.optimizer = tf.train.AdamOptimizer  #self.optimizer = tf.train.GradientDescentOptimizer

        #NEED TO DO!!!!! for the encoder
        #self.rnn_inputs

    def add_placeholders(self):
        
        self.input_data = tf.placeholder(tf.int32, [None, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.max_summary_length = tf.reduce_max(self.summary_length, name='max_dec_len')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
        self.deterministic = tf.constant(False)

    def make_cell(self, rnn_size, keep_prob):
        
        cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)
        
        return cell
    
    def process_encoding_input(self):
        
        #Remove the last word id from each batch and concat the <go> to the begining of each batch
        ending = tf.strided_slice(self.targets, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab_to_int['<go>']), ending], 1)

        return dec_input

    ##TODO: need to define rnn_inputs beforehand..
    def add_encoder(self):
        enc_embed_input = tf.nn.embedding_lookup(self.embeddings, self.input_data)
        forward_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell(self.rnn_size, self.keep_probability) for _ in range(self.num_layers)])
        backward_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell(self.rnn_size, self.keep_probability) for _ in range(self.num_layers)])

        #self.enc_state is a tuple (output_state_fw, output_state_bw)
        enc_output, self.enc_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, enc_embed_input,
                                                                     self.text_length, dtype=tf.float32)
        
        #cocatenate the forward and backward outputs
        self.enc_output = tf.concat(enc_output,2)
    
    def add_decoder(self):
        
        start_token = self.vocab_to_int['<go>']
        end_token = self.vocab_to_int['<eos>']

        #dynamic batch size
        self.dynamic_batch_size = tf.shape(self.input_data)[0]

        #create the decoder embeddings and cell
        #embedding matrix [vocab_size, embedding_dim] , dec_input [batch_size, time_stamp]
        dec_input = self.process_encoding_input()
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings, dec_input) #dec_embed_input has a shape of (batch_size, time_stamps, embed_dim=300)
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell(self.rnn_size, self.keep_probability) for _ in range(self.num_layers)])

        if self.mode == 'inference' and self.beam_search:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(self.enc_output, multiplier=self.beam_length)
            sequence_length = tf.contrib.seq2seq.tile_batch(self.text_length, multiplier=self.beam_length)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(self.enc_state[0], multiplier=self.beam_length)
            batch = self.dynamic_batch_size*self.beam_length
        else:
            encoder_outputs = self.enc_output
            sequence_length = self.text_length
            encoder_final_state = self.enc_state[0]
            batch = self.dynamic_batch_size

        #create an attention mechanism, scaled luong attention
        #returns tensor [batch_size, alignment_size]
        attn_mech = tf.contrib.seq2seq.LuongAttention(
            num_units= self.attn_size,
            memory = encoder_outputs,
            memory_sequence_length=sequence_length,
            scale=True)

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell= dec_cell,
            attention_mechanism = attn_mech,
            attention_layer_size=self.attn_size)

        # Input projection layer to feed embedded inputs to the cell
        # ** Essential when use_residual=True to match input/output dims
        input_layer = layers_core.Dense(self.vocab_size,
                                        dtype=tf.float32,
                                        name='input_projection')

        #dense final layer
        output_layer = layers_core.Dense(self.vocab_size,
                                         use_bias=False,
                                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1),
                                         name='output_projection')

        start_tokens = tf.fill([self.dynamic_batch_size], start_token)
        attention_zero = dec_cell.zero_state(dtype=tf.float32, batch_size=batch)
        decoder_initial_state = attention_zero.clone(cell_state=encoder_final_state)

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32, name='masks')

        if self.mode == 'training':

            helper = tf.contrib.seq2seq.TrainingHelper(  #creating the training logits
                inputs=dec_embed_input,
                sequence_length=self.summary_length,
                time_major=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=output_layer)

            #output and state at each time-step
            #self.train_dec_outputs shape is (64, ?, 9138)
            self.train_dec_outputs, self.train_dec_last_state, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=False,
                impute_finished=True,
                swap_memory=True,
                maximum_iterations=self.max_summary_length)

            # dec_outputs: collections.namedtuple(rnn_outputs, sample_id)
            # dec_outputs.rnn_output: [batch_size x max(dec_sequence_len) x dec_vocab_size+2], tf.float32
            # dec_outputs.sample_id [batch_size], tf.int32

            # logits: [batch_size x max_dec_len x dec_vocab_size+1]
            logits = tf.identity(self.train_dec_outputs.rnn_output, name='train_logits')

            #loss function
            self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.targets, 
                                                               weights=masks, name='train_batch_loss')

            #tensorboard operations 
            tf.summary.scalar('epoch_loss', tf.reduce_mean(self.batch_loss))

            #prediction sample for validation
            #self.train_predictions = tf.identity(self.train_dec_outputs.sample_id, name='training_ids')
            self.train_predictions = tf.expand_dims(self.train_dec_outputs.sample_id, -1, name='training_ids')

            #get training variables
            #self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        elif self.mode == 'inference':

            if not self.beam_search:

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embeddings,
                    start_tokens=start_tokens,
                    end_token=end_token)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=output_layer)
            else:

                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=dec_cell,
                    embedding=self.embeddings,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_length,
                    output_layer=output_layer,
                    length_penalty_weight=0.0)

            #def embed_and_input_proj(inputs):
               # return input_layer(tf.nn.embedding_lookup(self.embeddings, inputs))

            self.infer_dec_outputs, self.infer_dec_last_state, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=self.max_summary_length)

            # logits: [batch_size x max_dec_len x dec_vocab_size+1]
            #logits = self.infer_dec_outputs.predicted_ids

            if not self.beam_search:
                logits = tf.identity(self.infer_dec_outputs.rnn_output, name='prediction_logits')
                self.valid_predictions = tf.identity(self.infer_dec_outputs.sample_id, name='prediction_ids')
            else:
                logits = tf.no_op()
                self.valid_predictions = tf.identity(self.infer_dec_outputs.predicted_ids, name="prediction_ids")

            #loss function
            #self.batch_loss = tf.contrib.seq2seq.sequence_loss(targets=self.targets, weights=masks, logits=logits)

            #get training variables
            #self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def get_batches(self, sorted_texts, sorted_summaries):

        pad = self.vocab_to_int["<pad>"]
    
        #batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(sorted_texts)// self.batch_size):
            start_i = batch_i * self.batch_size
            summaries_batch = np.array(sorted_summaries[start_i:start_i + self.batch_size])
            texts_batch = np.array(sorted_texts[start_i:start_i + self.batch_size])

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in summaries_batch:
                pad_summaries_lengths.append(np.count_nonzero(summary != pad))

            pad_texts_lengths = []
            for text in texts_batch:
                pad_texts_lengths.append(np.count_nonzero(text != pad))

            yield summaries_batch, texts_batch, pad_summaries_lengths, pad_texts_lengths
            
    def add_training_optimizer(self):

        """
        This method simply combines calls compute_gradients() and apply_gradients() in place of minimize() if you want
        to process the gradient before applying them

        :return: Apply gradients to variables
        """

        if not self.adaptive_optimizer == "True":
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step*self.dynamic_batch_size,
                                                            100000, self.learning_rate_decay, staircase=True)
        else:
            self.learning_rate = self.starter_learning_rate

        optimizer = self.optimizer(self.learning_rate, name='training_op')  #gradient clipping implemented
        gradients = optimizer.compute_gradients(self.batch_loss)
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]

        if not self.adaptive_optimizer == "True":
            self.training_op = optimizer.apply_gradients(capped_gradients, global_step=self.global_step)
        else:
            self.training_op = optimizer.apply_gradients(capped_gradients)


    def save(self, sess, var_list=None, save_path=None):
        print('Saving model at {}'.format(save_path))
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        saver = tf.train.Saver(var_list)
        saver.save(sess, save_path, write_meta_graph=True)

    def restore(self, sess, var_list=None, ckpt_path=None):
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        self.restorer = tf.train.Saver(var_list)
        self.restorer.restore(sess, ckpt_path)
        print('Restore Finished!')

    def summary(self):
        summary_writer = tf.summary.FileWriter(
            logdir=self.checkpoint,
            graph=tf.get_default_graph())

    def build(self):
        self.add_placeholders()
        self.add_encoder()
        self.add_decoder()
        
    def train(self, sess, data_tuple, from_scratch=False,
              load_ckpt=None, save_path=None):
        
        #check for checkpoint
        if from_scratch is False and os.path.isfile(load_ckpt):
            self.restore(sess, load_ckpt)
            
        #add optimizer to graph
        self.add_training_optimizer()
        
        #initilize global variables
        sess.run(tf.global_variables_initializer())
        
        texts, summaries = data_tuple

        #implmenting k-fold cross validation - creating training and validation sets
        kfold = KFold(n_splits=self.fold, shuffle=True, random_state=11)
        
        update_loss_train = 0 
        batch_loss_train = 0
        summary_update_loss_train = [] # Record the update losses for saving improvements in the model

        update_check = (len(summaries)//self.batch_size//self.per_epoch)-1 
        
        for e in tqdm(range(self.epochs)):

            update_loss_train = 0
            batch_loss_train = 0
            output_tuple_data = []

            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                self.get_batches(texts, summaries)):

                start_time = time.time()

                # Evaluate 3 ops in the graph
                # => valid_predictions, loss, training_op(optimzier)
                batch_preds, batch_loss, _ = sess.run(

                    [self.train_predictions, self.batch_loss, self.training_op],
                    feed_dict={
                        self.input_data: texts_batch,
                        self.targets: summaries_batch,
                        self.summary_length: summaries_lengths,
                        self.keep_prob: self.keep_probability,
                        self.text_length: texts_lengths,
                    })

                batch_loss_train += batch_loss
                update_loss_train += batch_loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % self.display_step == 0 and batch_i > 0:

                    output_tuple = (e, self.epochs, batch_i, len(texts) // self.batch_size,
                                    batch_loss_train / self.display_step, batch_time*self.display_step)
                    output_tuple_data.append(output_tuple)

                    print('Train_Epoch:{:>3}/{}    Train_Batch:{:>4}/{}    Train_Loss:{:>6.3f}   Seconds:{:>4.2f}'
                              .format(e,
                                      self.epochs, 
                                      batch_i, 
                                      len(texts) // self.batch_size, 
                                      batch_loss_train / self.display_step, 
                                      batch_time*self.display_step))
                    batch_loss_train = 0
            
                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss_train/update_check,3))
                    summary_update_loss_train.append(update_loss_train)

                    #if the update loss is at a new minimum, save the model
                    if update_loss_train <= min(summary_update_loss_train):
                        print('New Record!') 
                        self.stop_early = 0
                        saver = tf.train.Saver() 
                        saver.save(sess, self.checkpoint)
                    else:
                        print("No Improvement.")
                        self.stop_early += 1
                        if self.stop_early == self.stop:
                            break
                    update_loss_train = 0

            Loading.save_pickle(os.path.join(os.path.dirname(self.checkpoint), 'data/output_data_training_epoch_{}.txt'.format(e)), output_tuple_data)

        if save_path:
            self.save(sess, save_path=save_path)
            
    def inference(self, sess, data_tuple, load_ckpt):
        
        self.restore(sess, ckpt_path=load_ckpt)
        texts, summaries = data_tuple
        
        for e in tqdm(range(self.epochs)):

            output_tuple_data = []
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                self.get_batches(texts, summaries)):

                start_time = time.time()
                batch_preds = sess.run(
                    [self.valid_predictions],
                    feed_dict={
                        self.input_data: texts_batch,
                        self.targets: summaries_batch,
                        self.summary_length: summaries_lengths,
                        self.keep_prob: self.keep_probability,
                        self.text_length: texts_lengths,
                    })

                end_time = time.time()
                batch_time = end_time - start_time

                batch_preds = np.array(batch_preds)
                batch_preds = np.squeeze(batch_preds, axis=0)

                #batch_preds shape: (64, 5, 10) [batch_size, time, beam_width]
                batch_preds = batch_preds.transpose([2, 0, 1])[0]

                for target_sent, input_sent, pred in zip(summaries_batch, texts_batch, batch_preds):

                    pad = self.vocab_to_int["<pad>"]
                    pred = list(pred)

                    actual_sent = ' '.join([self.int_to_vocab[index] for index in input_sent if index != pad])
                    actual_title = ' '.join([self.int_to_vocab[index] for index in target_sent if index != pad])

                    if not self.beam_search:
                        predicted_title = ' '.join([self.int_to_vocab[index] for index in pred if index != pad])  #beam search output in [batch_size, time, beam_width] shape.
                    else:
                        predicted_title = ' '.join([self.int_to_vocab[index] for index in pred if index != pad])  #beam search output in [batch_size, time, beam_width] shape.

                    output_tuple = (actual_sent, actual_title, predicted_title)
                    output_tuple_data.append(output_tuple)

            Loading.save_pickle(os.path.join(os.path.dirname(self.checkpoint), 'data/output_data_inference_epoch_{}.txt'.format(e)), output_tuple_data)