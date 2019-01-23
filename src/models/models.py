from utils import solve_eq_string, is_same_result, get_real_answer, is_number
import pandas as pd
from text_to_template import number_parsing, test_number_parsing
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, RepeatVector, Input
from keras.layers import recurrent, Bidirectional, LSTM, TimeDistributed, Flatten
from keras.models import Sequential, Model
from keras.utils import to_categorical
import numpy as np
from keras.layers import add
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json


class ExampleModel():
    def __init__(self, **kwargs):
        pass

    def fit(self, df, y=None):

        # TODO
        raise NotImplementedError
        return self

    def predict(self, df):
        # TODO
        raise NotImplementedError
        return corpus_df

    def score(self, corpus_df, frac=1, verbose=False, use_ans=True, output_errors=False):

        def solve(problem):
            try:
                a = solve_eq_string(problem["predicted_equations"], integer_flag=is_number(problem["text"]))
                return a
            except Exception as e:
                return []

        corpus_df = self.predict(corpus_df)

        error_list = []
        correct, total = 0, 0
        for k, problem in corpus_df.sample(frac=frac).iterrows():
            pred_ans = solve(problem)

            if is_same_result([problem['ans_simple']], pred_ans):
                correct += 1
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'], '1')]
            elif use_ans and is_same_result(get_real_answer(problem), pred_ans):
                correct += 1
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'], '1')]
            else:
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'], '0')]

            total += 1
            if verbose: print(correct, total, correct / total)

        if output_errors:
            return correct / total, pd.DataFrame(error_list,
                                                 columns=['ind', 'equations', 'predicted_equations', 'text', 'correct'])
        else:
            return correct / total


class DiltonModel(ExampleModel):
    def __init__(self, **kwargs):
        super(DiltonModel, self).__init__(**kwargs)
        self.hp = {  # Hyper-parameters
            'RNN': recurrent.GRU,
            'EMBED_HIDDEN_SIZE': 30,
            'SENT_HIDDEN_SIZE': 100,
            'BATCH_SIZE': 32,
            'EPOCHS': 40,
        }
        self.args = kwargs
        self.model = self.build()
        self.compile()

    def build(self):
        txtrnn = Sequential()
        txtrnn.add(Embedding(self.args['vocab_size'], self.hp['EMBED_HIDDEN_SIZE'],
                             input_length=self.args['story_maxlen']))
        txtrnn.add(Dropout(0.3))
        txtrnn.add(self.hp['RNN'](self.hp['SENT_HIDDEN_SIZE'], return_sequences=False))

        model = Sequential()
        model.add(self.hp['RNN'](self.hp['EMBED_HIDDEN_SIZE'], return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(self.args['vocab_answer_size'], activation='softmax'))

        return Model(txtrnn.input, model(txtrnn.output))

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, df, y=None):
        df['vars'] = df['ans_simple'].apply(lambda row: len(row))
        df['padded_ans'] = df['ans_simple'].apply(lambda row: (row+(max_vars-len(row))*[0][:(max_vars-len(row))]))
        df['full_ans'] = df.apply(lambda row: [row['vars']]+row['padded_ans'], axis=1)

class EncoderDecoder_model(ExampleModel):
    def __init__(self, **kwargs):
        super(EncoderDecoder_model, self).__init__(**kwargs)
        self.args = kwargs
        self.model = self.build()
        print(self.model.summary())
        self.compile()

    def build(self):
        inputs = Input(shape=(self.args['input_shape'],))
        # outputs = Dense(self.args['output_shape'], activation='softmax')(inputs)
        encoder1 = Embedding(self.args['txt_vocab_size'], 128)(inputs)
        encoder2 = Bidirectional(LSTM(128))(encoder1)
        encoder3 = RepeatVector(self.args['output_shape'])(encoder2)
        # decoder output model
        decoder1 = LSTM(128, return_sequences=True)(encoder3)
        # outputs = Dense(self.args['eqn_vocab_size'], activation='softmax')(decoder1)
        outputs = TimeDistributed(Dense(self.args['eqn_vocab_size'], activation='softmax'))(decoder1)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, df, y=None):
        X = np.vstack(df['X'])
        y = to_categorical(np.vstack(df['y']), self.args['eqn_vocab_size'])
        self.model.fit(x=X, y=y, epochs=100)

    def predict(self, df):
        X = np.vstack(df['X'])
        pred_matrix = self.model.predict(X)
        var_col = []
        eqn_col = []

        def get_value_from_vocab(arr):
            eqn_reversed_vocab = dict(zip(self.args['eqn_vocab'].values(), self.args['eqn_vocab'].keys()))
            l = []
            for a in arr:
                if a!=0:
                    l.append(eqn_reversed_vocab[a])
                else:
                    l.append(None)
            return l

        for i in range(pred_matrix.shape[0]):
            p = pred_matrix[i,:,:]
            p_vec = np.argmax(p,axis=1)
            len_var = p_vec[0]
            len_eqn = p_vec[1+self.args['var_length']]

            p_val = get_value_from_vocab(p_vec)
            var_vec = p_val[1:1+len_var]
            eqn_vec = p_val[2+len_var:2+len_var+len_eqn]
            var_col.append(var_vec)
            eqn_col.append(eqn_vec)


        return pd.DataFrame({'var': var_col, 'eqn': eqn_col})

