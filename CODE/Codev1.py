
# coding: utf-8

# In[105]:


import pytumblr
from keras.layers import recurrent,Bidirectional
import argparse
import re
from collections import Counter
import numpy as np
from numpy import zeros as np_zeros
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, LambdaCallback
from numpy.random import seed as random_seed
from numpy.random import randint as random_randint
from numpy.random import choice as random_choice
from numpy.random import randint as random_randint
from numpy.random import shuffle as random_shuffle
from numpy.random import rand
import os
import pickle
import logging
from time import time
import sys
#from kitchen.text.converters import getwriter


# In[106]:


os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

LOGGER = logging.getLogger(__name__) # Every log will use the module name
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.DEBUG)

DATASET_FILENAME = 'train/trainfile.txt'
test_set_fraction=0.1
NUMBER_OF_EPOCHS = 30
RNN = recurrent.LSTM
INPUT_LAYERS = 1
OUTPUT_LAYERS = 1
AMOUNT_OF_DROPOUT = 0.5
BATCH_SIZE = 50
SAMPLES_PER_EPOCH = 1000
HIDDEN_SIZE = 512
INITIALIZATION = "he_normal"  # : Gaussian initialization scaled by fan_in (He et al., 2014)
NUMBER_OF_CHARS = 100  # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
INVERTED = True
MODEL_CHECKPOINT_DIRECTORYNAME = 'models'
MODEL_CHECKPOINT_FILENAME = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
MODEL_DATASET_PARAMS_FILENAME = 'dataset_params.pickle'
MODEL_STARTING_CHECKPOINT_FILENAME = 'weights.hdf5'
CSV_LOG_FILENAME = 'log.csv'
# Parameters for the model and dataset
MAX_INPUT_LEN = 40
MIN_INPUT_LEN = 3
AMOUNT_OF_NOISE = 0.2 / MAX_INPUT_LEN
NUMBER_OF_CHARS = 100  # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéè .")

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)  # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
                                        chr(768), chr(769), chr(832), chr(833), chr(2387), chr(5151),
                                        chr(5152), chr(65344), chr(8242)),
                                    re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(
                                    re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)),
                                re.UNICODE)


# In[107]:


def generate_model(output_len, chars=None):
    """Generate the model"""
    print('Build model...')
    chars = chars or CHARS
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(INPUT_LAYERS):
        model.add(Bidirectional(RNN(HIDDEN_SIZE, init=INITIALIZATION,
                                 return_sequences=True), input_shape=(None, len(chars))))
        model.add(Dropout(AMOUNT_OF_DROPOUT))
    # For the decoder's input, we repeat the encoded input for each time step
   # model.add(RepeatVector(output_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(OUTPUT_LAYERS):
        model.add(Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True, init=INITIALIZATION)))
        model.add(Dropout(AMOUNT_OF_DROPOUT))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), init=INITIALIZATION)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# In[108]:


class Colors(object):
    """For nicer printouts"""
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# In[109]:


def show_samples(model, dataset, epoch, logs, X_dev_batch, y_dev_batch):
    """Selects 10 samples from the dev set at random so we can visualize errors"""
    #UTF8Writer = getwriter('utf8')
    #sys.stdout = UTF8Writer(sys.stdout)
    #PYTHONIOENCODING=utf8
    for _ in range(10):
        ind = random_randint(0, len(X_dev_batch))
        row_X, row_y = X_dev_batch[np.array([ind])], y_dev_batch[np.array([ind])]
        preds = model.predict_classes(row_X, verbose=0)
        q = dataset.character_table.decode(row_X[0])
        correct = dataset.character_table.decode(row_y[0])
        guess = dataset.character_table.decode(preds[0], calc_argmax=False)

        #if INVERTED:
         #   print('Q', q[::-1])  # inverted back!
        #else:
         #   print('Q', q)

        #print('A', correct)
        #print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
        #print('---')
        
        with open("data/outFile.txt", "a", encoding = "utf-8") as out:
          if INVERTED:
            out.write('Q '+ q[::-1])  # inverted back!
          else:
            out.write('Q '+ q)
        with open("data/outFile.txt", "a", encoding = "utf-8") as out:
          out.write('A '+ correct)
          if correct == guess:
            out.write(Colors.ok + '?' + ' ' + guess)
          else:
            out.write(Colors.fail + '?' + ' ' + guess)
          #out.write(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
          out.write('---')




# In[110]:


def iterate_training(model, dataset, initial_epoch):
    """Iterative Training"""

    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_DIRECTORYNAME + '/' + MODEL_CHECKPOINT_FILENAME,
                                 save_best_only=True)
    tensorboard = TensorBoard()
    csv_logger = CSVLogger(CSV_LOG_FILENAME)

    X_dev_batch, y_dev_batch = next(dataset.dev_set_batch_generator(1000))
    show_samples_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: show_samples(model, dataset, epoch, logs, X_dev_batch, y_dev_batch))

    train_batch_generator = dataset.train_set_batch_generator(BATCH_SIZE)
    validation_batch_generator = dataset.dev_set_batch_generator(BATCH_SIZE)

    model.fit_generator(train_batch_generator,
                        steps_per_epoch=SAMPLES_PER_EPOCH/BATCH_SIZE,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_batch_generator,
                        validation_steps=SAMPLES_PER_EPOCH,
                        callbacks=[checkpoint, tensorboard, csv_logger, show_samples_callback],
                        verbose=1,
                        initial_epoch=initial_epoch)



# In[111]:


def save_dataset_params(dataset):
    params = { 'chars': dataset.chars, 'y_max_length': dataset.y_max_length }
    with open(MODEL_CHECKPOINT_DIRECTORYNAME + '/' + MODEL_DATASET_PARAMS_FILENAME, 'wb') as f:
        pickle.dump(params, f)


# In[112]:



def main_news():
    """Main"""
    checkpoint_filename=None
    dataset_params_filename=None
    initial_epoch=0
    dataset = DataSet(DATASET_FILENAME)

    if not os.path.exists(MODEL_CHECKPOINT_DIRECTORYNAME):
        os.makedirs(MODEL_CHECKPOINT_DIRECTORYNAME)

    if dataset_params_filename is not None:
        with open(dataset_params_filename, 'rb') as f:
            dataset_params = pickle.load(f)

        assert dataset_params['chars'] == dataset.chars
        assert dataset_params['y_max_length'] == dataset.y_max_length

    else:
        save_dataset_params(dataset)

    model = generate_model(dataset.y_max_length, dataset.chars)

    if checkpoint_filename is not None:
        model.load_weights(checkpoint_filename)

    iterate_training(model, dataset, initial_epoch)

#def read_news():
    #UTF8Writer = getwriter('utf8')
    #sys.stdout = UTF8Writer(sys.stdout)
 #   PYTHONIOENCODING=UTF-8
  #  print("Reading news")
   # news = open(DATASET_FILENAME, encoding='utf-8').read()
    #print("Read news")

    #lines = [line for line in news.split('\n')]
    #print("Read {} lines of input corpus".format(len(lines)))

    #lines = [clean_text(line) for line in lines]
    #print("Cleaned text")

    #counter = Counter()
    #for line in lines:
     #   counter += Counter(line)
    #most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
    #print("most popular chars are ready")
    #lines = [line for line in lines if line and not bool(set(line) - most_popular_chars)]
    #print("Left with {} lines of input corpus".format(len(lines)))

    #return lines


# In[113]:


class DataSet(object):
    """
    Loads news articles from a file, generates misspellings and vectorizes examples.
    """

    def __init__(self, dataset_filename, test_set_fraction=0.1, inverted=True):
        self.inverted = inverted

        news = self.read_news(dataset_filename)
        questions, answers = self.generate_examples(news)

        chars_answer = set.union(*(set(answer) for answer in answers))
        chars_question = set.union(*(set(question) for question in questions))
        self.chars = sorted(list(set.union(chars_answer, chars_question)))
        self.character_table = CharacterTable(self.chars)

        split_at = int(len(questions) * (1 - test_set_fraction))
        (self.questions_train, self.questions_dev) = (questions[:split_at], questions[split_at:])
        (self.answers_train, self.answers_dev) = (answers[:split_at], answers[split_at:])

        self.x_max_length = max(len(question) for question in questions)
        self.y_max_length = max(len(answer) for answer in answers)

        self.train_set_size = len(self.questions_train)
        self.dev_set_size = len(self.questions_dev)

        print("Completed pre-processing")

    def train_set_batch_generator(self, batch_size):
        return self.batch_generator(self.questions_train, self.answers_train, batch_size)

    def dev_set_batch_generator(self, batch_size):
        return self.batch_generator(self.questions_dev, self.answers_dev, batch_size)

    def batch_generator(self, questions, answers, batch_size):
        start_index = 0

        while True:
            questions_batch = []
            answers_batch = []

            while len(questions_batch) < batch_size:
                take = min(len(questions) - start_index, batch_size - len(questions_batch))

                questions_batch.extend(questions[start_index: start_index + take])
                answers_batch.extend(answers[start_index: start_index + take])

                start_index = (start_index + take) % len(questions)

            yield self.vectorize(questions_batch, answers_batch)

    def add_noise_to_string(self, a_string, amount_of_noise):
        """Add some artificial spelling mistakes to the string"""
        if rand() < amount_of_noise * len(a_string):
            # Replace a character with a random character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
        if rand() < amount_of_noise * len(a_string):
            # Delete a character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
        if len(a_string) < MAX_INPUT_LEN and rand() < amount_of_noise * len(a_string):
            # Add a random character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
        if rand() < amount_of_noise * len(a_string):
            # Transpose 2 characters
            random_char_position = random_randint(len(a_string) - 1)
            a_string = (a_string[:random_char_position] +
                        a_string[random_char_position + 1] +
                        a_string[random_char_position] +
                        a_string[random_char_position + 2:])
        return a_string

    def vectorize(self, questions, answers):
        """Vectorize the questions and expected answers"""

        assert len(questions) == len(answers)

        X = np_zeros((len(questions), self.x_max_length, self.character_table.size), dtype=np.bool)

        for i in range(len(questions)):
            sentence = questions[i]
            for j, c in enumerate(sentence):
                X[i, j, self.character_table.char_indices[c]] = 1

        y = np_zeros((len(answers), self.y_max_length, self.character_table.size), dtype=np.bool)

        for i in range(len(answers)):
            sentence = answers[i]
            for j, c in enumerate(sentence):
                y[i, j, self.character_table.char_indices[c]] = 1

        return X, y

    def clean_text(self, text):
        """Clean the text - remove unwanted chars, fold punctuation etc."""

        text = text.strip()
        text = NORMALIZE_WHITESPACE_REGEX.sub(' ', text)
        text = RE_DASH_FILTER.sub('-', text)
        text = RE_APOSTROPHE_FILTER.sub("'", text)
        text = RE_LEFT_PARENTH_FILTER.sub("(", text)
        text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
        text = RE_BASIC_CLEANER.sub('', text)

        return text

    def read_news(self, dataset_filename):
        """Read the news corpus"""
        print("Reading news")
        news = open(dataset_filename, encoding='utf-8',errors="ignore").read()
        print("Read news")

        lines = [line for line in news.split('\n')]
        #print(lines)#eliane
        print("Read {} lines of input corpus".format(len(lines)))

        lines = [self.clean_text(line) for line in lines]
        print("Cleaned text")
        #start eliane
        myfile=open("testing/file_cleaned.txt","w",encoding="utf-8",errors="ingore")
        for line in lines:
            myfile.write(line+"\n")
        myfile.close()
        #end eliane
        counter = Counter()
        for line in lines:
            counter += Counter(line)
        #print(counter.most_common(NUMBER_OF_CHARS))
        most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
        #print(most_popular_chars)
        print("most popular characters are ready")

        lines = [line for line in lines if line and not bool(set(line) - most_popular_chars)]
        print("Left with {} lines of input corpus".format(len(lines)))

        return lines

    def generate_examples(self, corpus):
        """Generate examples of misspellings"""

        print("Generating examples")

        questions, answers, seen_answers = [], [], set()

        while corpus:
            line = corpus.pop()
            
            while len(line) > MIN_INPUT_LEN:
                if len(line) <= MAX_INPUT_LEN:
                    answer = line
                    line = ""
                else:
                    #print(line)
                    
                    space_location = line.rfind(" ", MIN_INPUT_LEN, MAX_INPUT_LEN - 1)
                    #print(space_location)
                    if space_location > -1:
                        answer = line[:space_location]
                        line = line[len(answer) + 1:]
                    else:
                        space_location = line.rfind(" ")  # no limits this time
                        if space_location == -1:
                            break  # we are done with this line
                        else:
                            line = line[space_location + 1:]
                            continue

                if answer and answer in seen_answers:
                    continue

                seen_answers.add(answer)
                answers.append(answer)

        print('Shuffle')
        random_shuffle(answers)
        print("Shuffled")

        for answer_index, answer in enumerate(answers):
            question = self.add_noise_to_string(answer, AMOUNT_OF_NOISE)
            question += '.' * (MAX_INPUT_LEN - len(question))
            answer += "." * (MAX_INPUT_LEN - len(answer))
            answers[answer_index] = answer
            assert len(answer) == MAX_INPUT_LEN

            question = question[::-1] if self.inverted else question
            questions.append(question)

        print("Generated questions and answers")

        return questions, answers



# In[114]:


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)

    def encode(self, C, maxlen):
        """Encode as one-hot"""
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool)  # pylint:disable=no-member
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)



# In[115]:


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Trains a deep spelling model.')
    #parser.add_argument('--checkpoint', type=str,
     #                   help='Filename of a model checkpoint to start the training from.')
    #parser.add_argument('--datasetparams', type=str,
     #                   help='Filename of a file with dataset params to load for continuing model training.')
    #parser.add_argument('initialepoch', type=int,
     #                   help='Initial epoch parameter for continuing model training.', default=0)

    #args = parser.parse_args()

    #main_news(args.checkpoint, args.datasetparams, args.initialepoch)
    main_news()
    #dataset = DataSet(DATASET_FILENAME)

