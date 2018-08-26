batch_size = 3  # Batch size for training.
epochs = 2  # Number of epochs to train for.
latent_dim = 30 #256  # Latent dimensionality of the encoding space.

# Path to the data txt file on disk.
train_data_path = 'rus_train.txt' # 'tmp' for tests
val_data_path = 'rus_val.txt'
enc_sent_size = 5
dec_sent_size = 5
enc_vec_size = 3
dec_vec_size = 3

steps_per_epoch = 10 # 0 # 301513//batch_size
ru_dict_source = 'softlink_ru'
en_dict_source = 'softlink_en'
max_nb_words = 1000000 
