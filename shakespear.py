import tensorflow as tf
import numpy as np
import os

text_file = tf.keras.utils.get_file('shakespeare.txt',
                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(text_file, 'rb').read()
text = text.decode(encoding='utf-8')
print('Total number of characters in the corpus is', len(text))
print('The first 100 characters of the corpus are as follows:\n', text[:1000])

vocab = sorted(set(text))
print('The number of unique characters in the corpus is', len(vocab))
print('A slice of the unique characters set :\n', vocab[:100])

# mapping unique characters to indices
char_idx = {u: i for i, u in enumerate(vocab)}

idx_char = np.array(vocab)

# vectorizing the text
text_int = np.array([char_idx[c] for c in text])

# creating a dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_int)
len_seq = 100
sequences = char_dataset.batch(len_seq + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# shuffle dataset and splitting it into 64 sentence batches
buffer = 1000
batch_size = 64

dataset = dataset.shuffle(buffer).batch(batch_size, drop_remainder=True)

# model building
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
         tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
         tf.keras.layers.Dense(vocab_size)
         ])
    return model


model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)
model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# to load our weights and save our training performance
check_dir = './training_checkpoints'

check_prefix = os.path.join(check_dir, "ckpt_{epoch}")

check_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_prefix, save_weights_only=True)

# training model
epoch = 2
his = model.fit(dataset, epochs=epoch, callbacks=[check_callback])

# generating new text

tf.train.latest_checkpoint(check_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(check_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# function to prepare out input for the model
def generate_text(model, num_generate, temperature, start_string):
    input_eval=[char_idx[s] for s in start_string]
    input_eval=tf.expand_dims(input_eval, 0)
    text_generated=[]
    model.reset_states()

    for i in range(num_generate): #running loop for number of char to generate
        predictions=model(input_eval)  #predicting single char
        predictions=tf.squeeze(predictions,0) #remove the batch dim

        # using a categorical distribution to predict the character returned by the model
        # higher temperature increases the probability of selecting a less likely character
        # lower --> more predictable

        predictions=predictions/temperature
        predicted_id=tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

        # The predicted character as the next input to the model
        # along with the previous hidden state
        # So the model makes the next prediction based on the previous character
        input_eval=tf.expand_dims([predicted_id],0)
        text_generated.append(idx_char[predicted_id])

    return start_string + ''.join(text_generated)


generated_text=generate_text(model, num_generate=500,temperature=1,start_string=u"ROMEO")
print(generated_text)