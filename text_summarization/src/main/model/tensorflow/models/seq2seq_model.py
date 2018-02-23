#!usr/bin/env/python

import os, time
import numpy as np
import tensorflow as tf
from main.model.tensorflow.decoders.beam_search_decoder import BeamSearchDecoder
from main.model.tensorflow.inference.beam_search import BeamSearchConfig
from main.model.tensorflow.models.model_base import ModelBase
from main.resources.helper_function import Loading
from sklearn.model_selection import KFold
from tqdm import tqdm


class BasicSeq2Seq(ModelBase):

    def __init__(self, params, mode, output_dir, name="base_seq2seq"):
        super(BasicSeq2Seq, self).__init__(params, mode, output_dir)

    def _add_placeholders(self):

        self.input_data = tf.placeholder(tf.int32, [None, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.max_summary_length = tf.reduce_max(self.summary_length, name='max_dec_len')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
        self.deterministic = tf.constant(False)

    @classmethod
    def _get_text(self, keys, data):
        sorted_texts = [v for d in data for k, v in d.items() if k == keys[0]]
        sorted_summaries = [v for d in data for k, v in d.items() if k == keys[1]]
        return sorted_texts, sorted_summaries

    def _use_beam_search(self):
        """Returns true iff the model should perform beam search."""
        return self.params["beam_length"] > 1

    def _batch_size(self):
        """Returns the dynamic batch size for a batch of examples"""
        return tf.shape(self.input_data)[0]

    def _make_cell(self, rnn_size):
        cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.params["dropout_input_keep_prob"],
                                      output_keep_prob=self.params["dropout_output_keep_prob"])
        return cell

    ##TODO: need to define rnn_inputs beforehand..
    def _add_encoder(self):
        enc_embed_input = tf.nn.embedding_lookup(self.embed_matrix, self.input_data)
        encoder_class = self.encoder_class(self.params, self.mode, self.output_dir)
        EncoderOutput = encoder_class.encode(enc_embed_input, self.text_length)
        self.enc_output, self.enc_final_state, attention_values, attention_values_length = EncoderOutput

    def _process_encoding_input(self):
        # Remove the last word id from each batch and concat the <go> to the begining of each batch
        ending = tf.strided_slice(self.targets, [0, 0], [self.params["batch_size"], -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.params["batch_size"], 1], self.word_to_ind['<go>']), ending], 1)
        return dec_input

    def _create_decoder_input(self):
        """create the decoder embeddings and cell embedding matrix [vocab_size, embedding_dim] ,
        dec_input [batch_size, time_stamp]"""
        dec_input = self._process_encoding_input()
        dec_embed_input = tf.nn.embedding_lookup(self.embed_matrix, dec_input)  # dec_embed_input has a shape of (batch_size, time_stamps, embed_dim=300)
        return dec_embed_input

    def _create_decoder_cell(self, enc_output, seq_length):
        return tf.contrib.rnn.MultiRNNCell([self._make_cell(self.params["rnn_size"]) for _ in range(self.params["num_layers"])])

    #TODO: look into length_penalty_weight, choose_successors_fn
    def _get_beam_search_decoder(self, decoder):
        """Wraps a decoder into a Beam Search decoder.

        Args:
          decoder: The original decoder

        Returns:
          A BeamSearchDecoder with the same interfaces as the original decoder.
        """

        config = BeamSearchConfig(
            beam_width=self.params["beam_length"],
            length_penalty_weight=self.params["inference.beam_search.length_penalty_weight"],
            start_token=self.start_tokens,
            end_token=self.end_token,
            embed_matrix=self.embed_matrix,
            vocab_size=len(self.word_to_ind)+1,
            summ_length = self.summary_length,
            max_length=self.max_summary_length)

        return BeamSearchDecoder(decoder=decoder, config=config)

    def _add_decoder(self):

        dynamic_batch_size = self._batch_size();
        self.start_tokens = tf.fill([dynamic_batch_size], self.word_to_ind['<go>'])
        self.end_token = self.word_to_ind['<eos>']

        if self.mode == 'predict' and self.params["use_beam"]:
            enc_output = tf.contrib.seq2seq.tile_batch(self.enc_output, multiplier=self.params["beam_length"])
            seq_length = tf.contrib.seq2seq.tile_batch(self.text_length, multiplier=self.params["beam_length"])
            enc_final_state = tf.contrib.seq2seq.tile_batch(self.enc_final_state[0], multiplier=self.params["beam_length"])
            dynamic_batch_size = dynamic_batch_size * self.params["beam_length"]
        else:
            enc_output = self.enc_output
            seq_length = self.text_length
            enc_final_state = self.enc_final_state[0]

        dec_embed_input = self._create_decoder_input()
        dec_cell = self._create_decoder_cell(enc_output, seq_length)
        decoder_instance = self._create_decoder(self.summary_length, self.max_summary_length)

        if not self.mode == "train" and self.params["use_beam"]:
            decoder_instance = self._get_beam_search_decoder(decoder_instance)

        dec_param_list = [enc_output, seq_length, enc_final_state, dynamic_batch_size]
        dec_outputs, dec_last_state, masks, logits, self.predictions = \
            decoder_instance._build_decoder(dec_cell, dec_embed_input, dec_param_list)

        # dec_outputs: collections.namedtuple(rnn_outputs, sample_id)
        # dec_outputs.rnn_output: [batch_size x max(dec_sequence_len) x dec_vocab_size+2], tf.float32
        # dec_outputs.sample_id [batch_size], tf.int32

        # loss function
        if self.mode == "train":
            self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.targets,
                                                               weights=masks, name='train_batch_loss')

            # tensorboard operations
            tf.summary.scalar('epoch_loss', tf.reduce_mean(self.batch_loss))

    def _build(self):
        self._add_placeholders()
        self._add_encoder()
        self._add_decoder()

    def get_batches(self, sorted_texts, sorted_summaries):

        pad = self.word_to_ind["<pad>"]

        # batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(sorted_texts) //  self.params["batch_size"]):
            start_i = batch_i *  self.params["batch_size"]
            summaries_batch = np.array(sorted_summaries[start_i:start_i +  self.params["batch_size"]])
            texts_batch = np.array(sorted_texts[start_i:start_i + self.params["batch_size"]])

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in summaries_batch:
                pad_summaries_lengths.append(np.count_nonzero(summary != pad))

            pad_texts_lengths = []
            for text in texts_batch:
                pad_texts_lengths.append(np.count_nonzero(text != pad))

            yield summaries_batch, texts_batch, pad_summaries_lengths, pad_texts_lengths

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

    def train(self, sess, data_tuple, from_scratch=False,
              load_ckpt=None, save_path=None):

        # check for checkpoint
        if from_scratch is False and os.path.isfile(load_ckpt):
            self.restore(sess, load_ckpt)

        # add optimizer to graph
        train_op = self._build_train_op()

        # initilize global variables
        sess.run(tf.global_variables_initializer())

        texts, summaries = data_tuple

        # implmenting k-fold cross validation - creating training and validation sets
        #kfold = KFold(n_splits=self.fold, shuffle=True, random_state=11)

        update_loss_train = 0
        batch_loss_train = 0
        summary_update_loss_train = []  # Record the update losses for saving improvements in the model

        update_check = (len(summaries) // self.params["batch_size"] // self.params["per_epoch"]) - 1

        for e in tqdm(range(self.params["per_epoch"])):

            update_loss_train = 0
            batch_loss_train = 0
            output_tuple_data = []

            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    self.get_batches(texts, summaries)):

                start_time = time.time()

                # Evaluate 3 ops in the graph
                # => valid_predictions, loss, training_op(optimzier)
                batch_preds, batch_loss, _ = sess.run(

                    [self.predictions, self.batch_loss, train_op],
                    feed_dict={
                        self.input_data: texts_batch,
                        self.targets: summaries_batch,
                        self.summary_length: summaries_lengths,
                        self.text_length: texts_lengths,
                    })

                batch_loss_train += batch_loss
                update_loss_train += batch_loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % self.params["display_step"] == 0 and batch_i > 0:
                    output_tuple = (e, self.params["per_epoch"], batch_i, len(texts) //  self.params["batch_size"],
                                    batch_loss_train / self.params["display_step"], batch_time * self.params["display_step"])
                    output_tuple_data.append(output_tuple)

                    print('Train_Epoch:{:>3}/{}    Train_Batch:{:>4}/{}    Train_Loss:{:>6.3f}   Seconds:{:>4.2f}'
                          .format(e,
                                  self.params["per_epoch"],
                                  batch_i,
                                  len(texts) //  self.params["batch_size"],
                                  batch_loss_train / self.params["display_step"],
                                  batch_time * self.params["display_step"]))
                    batch_loss_train = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss_train / update_check, 3))
                    summary_update_loss_train.append(update_loss_train)

                    # if the update loss is at a new minimum, save the model
                    if update_loss_train <= min(summary_update_loss_train):
                        print('New Record!')
                        self.params["stop_early"] = 0
                        saver = tf.train.Saver()
                        saver.save(sess, self.checkpoint)
                    else:
                        print("No Improvement.")
                        self.params["stop_early"] += 1
                        if self.params["stop_early"] == self.params["stop_update"]:
                            break
                    update_loss_train = 0

            Loading.save_pickle(
                os.path.join(os.path.dirname(self.checkpoint), 'data/output_data_training_epoch_{}.txt'.format(e)),
                output_tuple_data)

        if save_path:
            self.save(sess, save_path=save_path)

    def inference(self, sess, data_tuple, load_ckpt):

        self.restore(sess, ckpt_path=load_ckpt)

        texts, summaries = data_tuple

        for e in tqdm(range(self.params["per_epoch"])):

            output_tuple_data = []
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    self.get_batches(texts, summaries)):

                start_time = time.time()
                batch_preds = sess.run(
                    [self.predictions],
                    feed_dict={
                        self.input_data: texts_batch,
                        self.targets: summaries_batch,
                        self.summary_length: summaries_lengths,
                        self.text_length: texts_lengths,
                    })

                end_time = time.time()
                batch_time = end_time - start_time

                batch_preds = np.array(batch_preds)
                batch_preds = np.squeeze(batch_preds, axis=0)

                # batch_preds shape: (64, 5, 10) [batch_size, time, beam_width]
                batch_preds = batch_preds.transpose([2, 0, 1])[0]

                c = 0

                for target_sent, input_sent, pred in zip(summaries_batch, texts_batch, batch_preds):

                    pad = self.word_to_ind["<pad>"]
                    pred = list(pred)

                    actual_sent = ' '.join([self.ind_to_word[index] for index in input_sent if index != pad])
                    actual_title = ' '.join([self.ind_to_word[index] for index in target_sent if index != pad])

                    if not self.params["use_beam"]:
                        predicted_title = ' '.join([self.ind_to_word[index] for index in pred if
                                                    index != pad])  # beam search output in [batch_size, time, beam_width] shape.
                        print(actual_title); print(predicted_title)
                    else:
                        predicted_title = ' '.join([self.ind_to_word[index] for index in pred if
                                                    index != pad])  # beam search output in [batch_size, time, beam_width] shape.
                        print("----");
                        print('Actual sentence:  %s \n ' % actual_sent);
                        print('Actual title: %s \n ' % actual_title);
                        print('Prediction: %s ' % predicted_title);
                        # print(predicted_title2);print(predicted_title3)
                    print(c)
                    c += 1

                    output_tuple = (actual_sent, actual_title, predicted_title)
                    output_tuple_data.append(output_tuple)

            Loading.save_pickle(
                os.path.join(os.path.dirname(self.checkpoint), 'data/output_data_inference_epoch_{}.txt'.format(e)),
                output_tuple_data)