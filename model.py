import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Reset the graph to ensure that it is ready for training
# tf.compat.v1.reset_default_graph()
tf.reset_default_graph()


class LSTM_MODEL(object):
    def __init__(self, max_words, vocab2ints):  #ok
        # Set the Hyperparameters
        # self.epochs = 100
        self.max_words = max_words
        self.vocab2ints = vocab2ints
        self.batch_size = 128
        self.learning_rate = 0.005
        self.keep_probability = 0.75
        
        self.num_layers = 2
        self.rnn_size = 512
        self.encoding_embedding_size = 512
        self.decoding_embedding_size = 512
       
        self.checkpoint = "best_model.ckpt" 
        

        self.sequence_length = tf.placeholder_with_default(self.max_words, None, name='sequence_length')   #it is equal to max_words for each batch   
        self.input_data, self.targets, self.lr, self.keep_prob = self.get_placeholders()  #load model inputs
       
        training_logits, inference_logits = self.seq2seq_model(tf.reverse(self.input_data, [-1]))  

   
        tf.identity(inference_logits, 'logits')  #create a tensor for the inference logits, required if loading a checkpoint version of the model

        with tf.name_scope("optimization"):
            self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, tf.ones([tf.shape(self.input_data)[0], self.sequence_length])) # loss fnc

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

            self.gradients = self.optimizer.compute_gradients(self.cost)
            self.capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.gradients if grad is not None]   # gradient clipping
            self.train_optimizer = self.optimizer.apply_gradients(self.capped_gradients)


    
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


        

    def get_placeholders(self): 
        '''Create palceholders for inputs to the model'''

        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return input_data, targets, lr, keep_prob

    
    def process_encoded_input(self):  
        end = tf.strided_slice(self.targets, [0, 0], [self.batch_size, -1], [1, 1])  #remove the last word id from batch
        new_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab2ints['<GO>']), end], 1)  #concat '<GO>' to the begining of batch

        return new_input
    

    def encoding_layer(self, embeded_inputs):  
        '''build encoding layer'''
        lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.keep_prob)
        encoded_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_layers)
        _, encoded_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoded_cell,
                                                    cell_bw = encoded_cell,
                                                    sequence_length = self.sequence_length,
                                                    inputs = embeded_inputs, 
                                                    dtype=tf.float32)
        return encoded_state

    def train_decoded_layers(self, encoder_state, decoder_cell, embeded_input, scope, output_fn): 
        '''Decode the training data'''
        
        attention_states = tf.zeros([self.batch_size, 1, decoder_cell.output_size])
        keys, vals, score_fn, construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau",
                                                                                                  num_units = decoder_cell.output_size)
        
        train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], keys, vals, score_fn, construct_fn, name = "attn_dec_train")
        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, train_decoder_fn, embeded_input, self.sequence_length, scope = scope)
        train_pred_drop = tf.nn.dropout(train_pred, self.keep_prob)

        return output_fn(train_pred_drop)

   
    def infer_decoded_layers(self, encoder_state, decoder_cell, dec_embeddings, scope, output_fn): 
        '''Decode the prediction data'''
        start_of_sequence_id = self.vocab2ints['<GO>']
        end_of_sequence_id = self.vocab2ints['<EOS>']
        max_length = self.sequence_length - 1
        vocab_size = len(self.vocab2ints)

        attention_states = tf.zeros([self.batch_size, 1, decoder_cell.output_size])
        keys, vals, score_fn, construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau",
                                                                                                    num_units = decoder_cell.output_size)
        
        infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn, encoder_state[0], keys, vals, score_fn, construct_fn, 
                                                                            dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_length, vocab_size, 
                                                                            name = "attn_dec_inf")
        
        infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, infer_decoder_fn, scope = scope)
        
        return infer_logits

    def decoding_layer(self, embeded_input, dec_embeddings, encoder_state): 
        vocab_size = len(self.vocab2ints)

        with tf.variable_scope("decoding") as scope:
            lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.keep_prob)
            dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_layers)
            
            weights = tf.truncated_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()
            output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope = scope, weights_initializer = weights, biases_initializer = biases)

            training_logits = self.train_decoded_layers(encoder_state, dec_cell, embeded_input, scope, output_fn)                                   
            scope.reuse_variables()
            inference_logits = self.infer_decoded_layers(encoder_state, dec_cell, dec_embeddings, scope, output_fn)

        return training_logits, inference_logits

 
    def seq2seq_model(self, input_data):  
        vocab_size = len(self.vocab2ints)
        encoder_embeded_input = tf.contrib.layers.embed_sequence(input_data, 
                                                        vocab_size+1, 
                                                        self.encoding_embedding_size,
                                                        initializer = tf.random_uniform_initializer(0,1))
        encoder_state = self.encoding_layer(encoder_embeded_input)

        decoder_input = self.process_encoded_input()
        decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size+1, self.decoding_embedding_size], 0, 1))
        decoder_embeded_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
        
        training_logits, inference_logits = self.decoding_layer(decoder_embeded_input, 
                                                    decoder_embeddings, 
                                                    encoder_state)
        return training_logits, inference_logits


    def fit(self, questions_batch, answers_batch):
        _, loss = self.sess.run(
            [self.train_optimizer, self.cost],
            {self.input_data: questions_batch,
             self.targets: answers_batch,
             self.lr: self.learning_rate,
             self.sequence_length: answers_batch.shape[1],
             self.keep_prob: self.keep_probability})

        return loss

    def save(self):
        self.saver.save(self.sess, self.checkpoint)