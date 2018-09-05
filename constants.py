# random_case = False
# 
# batch_size = 5  # Batch size for training.
# epochs = 10  # Number of epochs to train for.
# latent_dim = 200 #256  # Latent dimensionality of the
# # encoding space.
# 
# # Path to the data txt file on disk.
# train_data_path = 'rus_train.txt' # 'tmp' for tests
# val_data_path = 'rus_val.txt'
# enc_sent_size = 15
# dec_sent_size = 15
# enc_vec_size = 100
# dec_vec_size = 300
# 
# steps_per_epoch =  301513 // (batch_size*2)
# ru_dict_source = 'softlink_ru'
# en_dict_source = 'softlink_en'
# max_nb_words = 100000
# 
path_checkpoint = 'checkpoint.keras'
# 
## test variant
batch_size = 5  # Batch size for training.
epochs = 3  # Number of epochs to train for.
latent_dim = 20 #256  # Latent dimensionality of the
# encoding space.

# Path to the data txt file on disk.
train_data_path = 'rus_train.txt' # 'tmp' for tests
val_data_path = 'rus_val.txt'
enc_sent_size = 10
dec_sent_size = 10
enc_vec_size = 5
dec_vec_size = 5
state_size = 20

# steps_per_epoch = 10000 # 301513 // (batch_size*2)
ru_dict_source = 'softlink_ru'
en_dict_source = 'softlink_en'
max_nb_words = 10000

start_word = 'SSTTAARRTT '
end_word = ' EENNDD'

random_case = True
