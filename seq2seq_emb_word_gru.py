import tensorflow as tf
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding
import numpy as np
from constants import *
from prepare_data import *
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.optimizers import RMSprop
from my_keras import *

num_words = 1000

mark_start = start_word.lower()
mark_end = end_word.lower()

input_texts, target_texts = get_data_seq2seq(train_data_path)

tokenizer_src = TokenizerWrap(texts=input_texts,
                              padding='pre',
                              reverse=True,
                              num_words=num_words)
    
tokenizer_dest = TokenizerWrap(texts=target_texts,
                               padding='post',
                               reverse=False,
                               num_words=num_words)

tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded

token_start = tokenizer_dest.word_index[mark_start.strip()]
token_end = tokenizer_dest.word_index[mark_end.strip()]

delim = 10000
encoder_input_data = tokens_src[:delim]
decoder_input_data = tokens_dest[:delim, :-1]
decoder_output_data = tokens_dest[:delim, 1:]

encoder_input = Input(shape=(None, ), name='encoder_input')
enc_embedding_matrix = np.random.uniform(-1,
                                         1,
                                         size = (num_words,
                                                 enc_vec_size))
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=enc_vec_size,
                              weights=[enc_embedding_matrix],
                              trainable=False)

encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=False)

def connect_encoder():
    net = encoder_input
    net = encoder_embedding(net)
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    encoder_output = net    
    return encoder_output

encoder_output = connect_encoder()

## decoder
dec_embedding_matrix = np.random.uniform(-1,
                                         1,
                                         size = (num_words,
                                                 dec_vec_size))
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=dec_vec_size,
                              weights=[dec_embedding_matrix],
                              trainable=False)

decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')

decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=dec_vec_size,
                              weights=[dec_embedding_matrix],
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(initial_state):
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)
model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])
decoder_output = connect_decoder(
    initial_state=decoder_initial_state)
model_decoder = Model(inputs=[decoder_input,
                              decoder_initial_state],
                      outputs=[decoder_output])


# optimize
optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])


x_data = \
{
    'encoder_input': encoder_input_data,
    'decoder_input': decoder_input_data
}

y_data = \
{
    'decoder_output': decoder_output_data
}


# validation_split = 10000 / len(enc_input_data)

model_train.fit(x=x_data,
                y=y_data,
                batch_size=batch_size,
                epochs=epochs)
                # validation_split=validation_split)


def translate(input_text, true_output_text=None):
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    
    initial_state = model_encoder.predict(input_tokens)
    max_tokens = tokenizer_dest.max_tokens
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }
        decoder_output = model_decoder.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer_dest.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
    output_tokens = decoder_input_data[0]
    
    # Print the input-text.
    print("Input text:")
    print(input_text)
    print()

    # Print the translated output-text.
    print("Translated text:")
    print(output_text)
    print()

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()


idx = 3
translate(input_text=input_texts[idx],
          true_output_text=target_texts[idx])
