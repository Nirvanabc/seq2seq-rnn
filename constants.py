batch_size = 50  # Batch size for training.
epochs = 5  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

# Path to the data txt file on disk.
train_data_path = 'rus_train.txt' # 'tmp' for tests
val_data_path = 'rus_val.txt'
enc_sent_size = 20
dec_sent_size = 20
enc_vec_size = 100
dec_vec_size = 300

steps_per_epoch = 300
ru_dict_source = 'softlink_ru'
en_dict_source = 'softlink_en'
max_nb_words = 1000000 
