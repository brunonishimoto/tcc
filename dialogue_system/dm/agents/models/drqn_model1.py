from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam


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

    def build_model(self):
        """Builds and returns model/graph of neural network."""
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.input_dim, activation=self.activation))
        model.add(LSTM(self.hidden_size, activation=self.activation))
        model.add(Dense(self.output_dim, activation=self.activation_out))
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr, decay=self.lr_decay))

        return model
