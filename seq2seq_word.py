import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
from constants import *
from prepare_data import *
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

input_texts, target_texts = get_data_seq2seq(train_data_path)
tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(target_texts)
word_index = tokenizer.word_index

nb_words = min(max_nb_words, len(word_index))+1

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, enc_vec_size))

encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, dec_vec_size))
decoder_lstm = LSTM(latent_dim, return_sequences=True,
                    return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(nb_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into
# `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training

test_batch_gen = next_batch_keras(input_texts,
                                  target_texts,
                                  batch_size,
                                  enc_vec_size,
                                  dec_vec_size,
                                  enc_sent_size,
                                  dec_sent_size,
                                  tokenizer)

# plot_model(model, to_file='model.png', show_shapes=True)

# here we see some magic, which comes from keras mistakes
# which are described in README 4.
def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

model.compile(optimizer='nadam',
              loss=sparse_cross_entropy,
              target_tensors=[decoder_target])

# magic ends

# Save model
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


model.fit_generator(test_batch_gen,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    verbose = 2)
                    # callbacks=[callback_checkpoint])



# restore the model
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h,
                         decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_string):
    input_seq = prepare_input_string(input_string)
    states_value = encoder_model.predict([input_seq])
    target_seq = np.zeros((1, 1, dec_vec_size))
    # target_seq[0, 0] = word2vec(dec_dict,
    #                             "ssttaarrtt",
    #                             dec_vec_size)
    target_seq[0, 0] = random.uniform(-1, 1,
                                      size = dec_vec_size)
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[
            sampled_token_index + 1]
        decoded_sentence += " " + sampled_word
        if (sampled_word == 'eenndd' or
            len(decoded_sentence.split()) > dec_sent_size):
            stop_condition = True
            
        # Update the target sequence (of length 1).
        # target_seq[0, 0] = word2vec(dec_dict,
        #                             sampled_word,
        #                             dec_vec_size)
        target_seq[0, 0] = random.uniform(-1, 1,
                                          size = dec_vec_size)
        # Update states
        states_value = [h, c]
        
    return decoded_sentence
        

for seq_index in range(10):
    # seq_index = 0
    input_string = input_texts[seq_index + 10000: seq_index + 10001][0]
    decoded_sentence = decode_sequence(input_string)
    print('-')
    print('Input sentence:', input_texts[seq_index + 100])
    print('Decoded sentence:', decoded_sentence)
