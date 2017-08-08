


class configuration(object):
    
    def __init__(self):
        
        # hyperparameters
        self.epochs = 30 # was 50, use 100
        self.batch_size = 64
        self.rnn_size = 256
        self.num_layers = 1
        self.learning_rate = 0.005
        self.keep_probability = 0.75

        self.learning_rate_decay = 0.95
        self.min_learning_rate = 0.0005
        self.display_step = 20 
        self.stop_early = 0 
        self.stop = 3 
        self.per_epoch = 3 
        
        #set optimizer
        self.optimizer = tf.train.AdamOptimizer

        # checkpoint path
        self.ckpt_dir = '/home/paperspace/Documents/development/models/tensorflow/best_model_TEST.ckpt'


class seq2seq(object):
    
    def __init__(self, configuration, *args):
        
        #additional required arguments
        self.embeddings = args[0]
        self.vocab_to_int = args[1]
        self.int_to_vocab = args[2]
        self.mode = args[3]
                        
        assert self.mode in ['training', 'evaluation', 'inference']

        self.batch_size = configuration.batch_size
        self.checkpoint = configuration.ckpt_dir
        self.display_step = configuration.display_step
        self.epochs = configuration.epochs
        self.keep_probability = configuration.keep_probability
        self.learning_rate = configuration.learning_rate
        self.learning_rate_decay = configuration.learning_rate_decay
        self.num_layers = configuration.num_layers
        self.min_learning_rate = configuration.min_learning_rate
        self.per_epoch = configuration.per_epoch
        self.rnn_size = configuration.rnn_size
        self.stop = configuration.stop
        self.stop_early = configuration.stop_early
        self.vocab_size = len(self.vocab_to_int)+1

        
        self.optimizer = configuration.optimizer
        
        #NEED TO DO!!!!! for the encoder
        #self.rnn_inputs

    def add_placeholders(self):
        
        self.input_data = tf.placeholder(tf.int32, [None, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.max_summary_length = tf.reduce_max(self.summary_length, name='max_dec_len')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
    
    def make_cell(self, rnn_size, keep_prob):
        
        cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)
        
        return cell
    
    def process_encoding_input(self):
        
        #Remove the last word id from each batch and concat the <GO> to the begining of each batch
        ending = tf.strided_slice(self.targets, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab_to_int['<go>']), ending], 1)

        return dec_input

    ##TODO: need to define rnn_inputs beforehand..
    def add_encoder(self):
        
        enc_embed_input = tf.nn.embedding_lookup(self.embeddings, self.input_data)
        forward_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell(self.rnn_size, self.keep_probability) for _ in range(self.num_layers)])
        backward_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell(self.rnn_size, self.keep_probability) for _ in range(self.num_layers)])
        enc_output, self.enc_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, enc_embed_input, self.text_length, 
                                                                dtype=tf.float32)
        
        #cocatenate the forward and backward outputs
        self.enc_output = tf.concat(enc_output,2)
    
    def add_decoder(self):
        
        start_token = self.vocab_to_int['<go>']
        end_token = self.vocab_to_int['<eos>']
        
        dec_input = self.process_encoding_input()
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings, dec_input)
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell(self.rnn_size, self.keep_probability) for _ in range(self.num_layers)])
        
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.rnn_size, 
            memory=self.enc_output, 
            memory_sequence_length=self.text_length, 
            normalize=False,
            name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
            cell=dec_cell, 
            attention_mechanism=attn_mech, 
            attention_size=self.rnn_size)
        
        output_layer = Dense(
            self.vocab_size, 
            kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
        initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
            cell_state=self.enc_state[0],
            attention=_zero_state_tensors(self.rnn_size, self.batch_size, tf.float32))   

        if self.mode == 'training':

            #creating the training logits
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=dec_embed_input, 
                sequence_length=self.summary_length,
                time_major=False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell, 
                helper=training_helper, 
                initial_state=initial_state, 
                output_layer=output_layer) 

            self.train_dec_outputs, self.train_dec_last_state = tf.contrib.seq2seq.dynamic_decode(
                training_decoder, 
                output_time_major=False,
                impute_finished=True, 
                maximum_iterations=self.max_summary_length)

            # dec_outputs: collections.namedtuple(rnn_outputs, sample_id)
            # dec_outputs.rnn_output: [batch_size x max(dec_sequence_len) x dec_vocab_size+2], tf.float32
            # dec_outputs.sample_id [batch_size], tf.int32

            # logits: [batch_size x max_dec_len x dec_vocab_size+1]
            logits = tf.identity(self.train_dec_outputs.rnn_output, 'logits')

            # Create the weights for sequence_loss
            masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32, name='masks')

            #loss function
            self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.targets, 
                                                               weights=masks, name='batch_loss')

            #tensorboard operations 
            tf.summary.scalar('epoch_loss', tf.reduce_mean(self.batch_loss))

            #prediction sample for validation
            self.valid_predictions = tf.identity(self.train_dec_outputs.sample_id, name='valid_preds')

            #get training variables
            #self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        elif self.mode == 'inference':

            start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size], name='start_tokens')

            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embeddings, 
                start_tokens=start_tokens, 
                end_token=end_token)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell, 
                helper=inference_helper, 
                initial_state=initial_state, 
                output_layer=output_layer)

            self.infer_dec_outputs, self.infer_dec_last_state = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder, 
                output_time_major=False,
                impute_finished=True, 
                maximum_iterations=self.max_summary_length)

            logits = tf.identity(self.infer_dec_outputs.sample_id, name='predictions')

            #get training variables
            #self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def pad_sentence_batch(self, sentence_batch):
        
        #Pad sentences with <pad> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        
        return [sentence + [self.vocab_to_int['<pad>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def get_batches(self, sorted_texts, sorted_summaries):
    
        #batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(sorted_texts)// self.batch_size):
            start_i = batch_i * self.batch_size
            summaries_batch = sorted_summaries[start_i:start_i + self.batch_size]
            texts_batch = sorted_texts[start_i:start_i + self.batch_size]
            pad_summaries_batch = np.array(self.pad_sentence_batch(summaries_batch))
            pad_texts_batch = np.array(self.pad_sentence_batch(texts_batch))

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))

            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))

            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths
            
    def add_training_optimizer(self):   
        optimizer = self.optimizer(self.learning_rate, name='training_op')  #gradient clipping implemented
        gradients = optimizer.compute_gradients(self.batch_loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.training_op = optimizer.apply_gradients(capped_gradients)

    def save(self, sess, var_list=None, save_path=None):
        print('Saving model at {}'.format(save_path))
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        saver = tf.train.Saver(var_list)
        saver.save(sess, save_path, write_meta_graph=False)

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
        
        update_loss_train = 0 
        batch_loss_train = 0
        update_loss_validate = 0 
        batch_loss_validate = 0
        summary_update_loss_train = [] # Record the update losses for saving improvements in the model
        summary_update_loss_validate = []

        
        update_check = (len(summaries)//self.batch_size//self.per_epoch)-1 
        
        for e in tqdm(range(self.epochs)):
            
            update_loss_train = 0
            batch_loss_train = 0
            
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                self.get_batches(texts, summaries)):
                
                start_time = time.time()
                
                # Evaluate 3 ops in the graph
                # => valid_predictions, loss, training_op(optimzier)
                batch_preds, batch_loss, _ = sess.run(
                    [self.valid_predictions, self.batch_loss, self.training_op],
                    feed_dict={
                        self.input_data: texts_batch,
                        self.targets: summaries_batch,
                        self.lr: self.learning_rate,
                        self.summary_length: summaries_lengths,
                        self.keep_prob: self.keep_probability,
                        self.text_length: texts_lengths,
                    })
              
                batch_loss_train += batch_loss
                update_loss_train += batch_loss
                end_time = time.time()
                batch_time = end_time - start_time  
                
                if batch_i % self.display_step == 0 and batch_i > 0:
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
                              
        if save_path:
            self.save(sess, save_path=save_path)
           

tf.reset_default_graph()
config = configuration()
model = seq2seq(config, word_embedding_matrix, vocab_to_int, int_to_vocab, 'training')
model.build()
model.summary()
print('Training model built!')

tf.reset_default_graph()     
with tf.Session() as sess:
    config = configuration()
    model = seq2seq(config, word_embedding_matrix, vocab_to_int, int_to_vocab, 'training')
    model.build()
    data = (X_train, y_train)
    loss_history = model.train(sess, data, from_scratch=True, 
                               save_path=model.checkpoint+f'epoch_{model.epochs}_attention')
    
