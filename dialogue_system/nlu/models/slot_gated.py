import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class SlotGated:
    def __init__(self, config):
        self.params = config["nlu"]
        self.model_type = self.params["model_type"]
        self.add_final_state_to_intent = True

        if self.model_type == "full":
            self.remove_slot_attn = False
        elif self.model_type == "intent_only":
            self.remove_slot_attn = True
        else:
            log(["debug"], "unknown model type")
            exit(1)

    def createModel(self, input_data, input_size, sequence_length, slot_size, intent_size, layer_size = 128, isTraining = True):
        cell_fw = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(layer_size)
        cell_bw = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(layer_size)

        if isTraining == True:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                                output_keep_prob=0.5)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                                output_keep_prob=0.5)

        embedding = tf.compat.v1.get_variable('embedding', [input_size, layer_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)

        state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32)

        final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
        state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
        state_shape = state_outputs.get_shape()

        with tf.compat.v1.variable_scope('attention'):
            slot_inputs = state_outputs
            if self.remove_slot_attn == False:
                with tf.compat.v1.variable_scope('slot_attn'):
                    attn_size = state_shape[2].value
                    origin_shape = tf.shape(state_outputs)
                    hidden = tf.expand_dims(state_outputs, 1)
                    hidden_conv = tf.expand_dims(state_outputs, 2)
                    # hidden shape = [batch, sentence length, 1, hidden size]
                    k = tf.compat.v1.get_variable("AttnW", [1, 1, attn_size, attn_size])
                    hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                    hidden_features = tf.reshape(hidden_features, origin_shape)
                    hidden_features = tf.expand_dims(hidden_features, 1)
                    v = tf.compat.v1.get_variable("AttnV", [attn_size])

                    slot_inputs_shape = tf.shape(slot_inputs)
                    slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
                    y = core_rnn_cell._linear(slot_inputs, attn_size, True)
                    y = tf.reshape(y, slot_inputs_shape)
                    y = tf.expand_dims(y, 2)
                    s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
                    a = tf.nn.softmax(s)
                    # a shape = [batch, input size, sentence length, 1]
                    a = tf.expand_dims(a, -1)
                    slot_d = tf.reduce_sum(a * hidden, [2])
            else:
                attn_size = state_shape[2].value
                slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

            intent_input = final_state
            with tf.compat.v1.variable_scope('intent_attn'):
                attn_size = state_shape[2].value
                hidden = tf.expand_dims(state_outputs, 2)
                k = tf.compat.v1.get_variable("AttnW", [1, 1, attn_size, attn_size])
                hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
                v = tf.compat.v1.get_variable("AttnV", [attn_size])

                y = core_rnn_cell._linear(intent_input, attn_size, True)
                y = tf.reshape(y, [-1, 1, 1, attn_size])
                s = tf.reduce_sum(v*tf.tanh(hidden_features + y), [2,3])
                a = tf.nn.softmax(s)
                a = tf.expand_dims(a, -1)
                a = tf.expand_dims(a, -1)
                d = tf.reduce_sum(a * hidden, [1, 2])

                if self.add_final_state_to_intent == True:
                    intent_output = tf.concat([d, intent_input], 1)
                else:
                    intent_output = d

            with tf.compat.v1.variable_scope('slot_gated'):
                intent_gate = core_rnn_cell._linear(intent_output, attn_size, True)
                intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[1].value])
                v1 = tf.compat.v1.get_variable("gateV", [attn_size])
                if self.remove_slot_attn == False:
                    slot_gate = v1 * tf.tanh(slot_d + intent_gate)
                else:
                    slot_gate = v1 * tf.tanh(state_outputs + intent_gate)
                slot_gate = tf.reduce_sum(slot_gate, [2])
                slot_gate = tf.expand_dims(slot_gate, -1)
                if self.remove_slot_attn == False:
                    slot_gate = slot_d * slot_gate
                else:
                    slot_gate = state_outputs * slot_gate
                slot_gate = tf.reshape(slot_gate, [-1, attn_size])
                slot_output = tf.concat([slot_gate, slot_inputs], 1)

        with tf.compat.v1.variable_scope('intent_proj'):
            intent = core_rnn_cell._linear(intent_output, intent_size, True)

        with tf.compat.v1.variable_scope('slot_proj'):
            slot = core_rnn_cell._linear(slot_output, slot_size, True)

        outputs = [slot, intent]
        return outputs
