from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten
from keras.optimizers import Adam
from keras.layers import concatenate


class DRQNModel1:

    def __init__(self, config):
        self.model_parameters = config['model']

        self.lr = self.model_parameters['learning_rate']
        self.lr_decay = self.model_parameters['lr_decay']
        self.hidden_size = self.model_parameters['dqn_hidden_size']
        self.input_dim = self.model_parameters['input_dim']
        self.activation = self.model_parameters['activation']
        self.activation_out = self.model_parameters['activation_out']
        self.loss = self.model_parameters['loss']
        self.output_dim = self.model_parameters['output_dim']
        self.db_size = self.model_parameters['db_size']

    def build_model(self):
        """Builds and returns model/graph of neural network."""
        observation1 = Input(shape=(1, self.input_dim[1]))
        observation2 = Input(shape=(1, self.input_dim[1]))

        encoded_observation1 = LSTM(self.hidden_size)(observation1)
        encoded_observation2 = LSTM(self.hidden_size)(observation2)

        merged_vector = concatenate([encoded_observation1, encoded_observation2])
        outputs = Dense(self.output_dim, activation=self.activation_out)(merged_vector)

        model = Model(input=[observation1, observation2], outputs=outputs)
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr, decay=self.lr_decay))

        return model
