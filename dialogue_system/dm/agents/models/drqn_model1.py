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
        observation = Input(shape=(self.input_dim[0], self.input_dim[1] - self.db_size[1]))
        db_input = Input(shape=self.db_size)

        encoded_observation = LSTM(self.hidden_size)(observation)
        db_attention = Dense(10, activation=self.activation)(db_input)
        db_flatten = Flatten()(db_attention)
        merged_vector = concatenate([encoded_observation, db_flatten])

        outputs = Dense(self.output_dim, activation=self.activation_out)(merged_vector)

        model = Model(input=[observation, db_input], outputs=outputs)
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr, decay=self.lr_decay))

        return model
