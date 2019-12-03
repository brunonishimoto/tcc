import os
import tensorflow as tf
import numpy as np
import dialogue_system.constants as const

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from .models import SlotGated
from utils.util import loadVocabulary
from utils.util import sentenceToIds, padSentence
from utils.util import log

class NLUSlotGated:
    def __init__(self, config):
        self.params = config["nlu"]
        self.model_type = self.params["model_type"]
        self.add_final_state_to_intent = True
        self.model_path = self.params["model_path"]
        self.vocab_path = self.params["vocab_path"]
        self.layer_size = self.params["layer_size"]

        if self.model_type == "full":
            self.remove_slot_attn = False
        elif self.model_type == "intent_only":
            self.remove_slot_attn = True
        else:
            log(["debug"], "unknown model type")
            exit(1)

        self.model = SlotGated(config)

        self.in_vocab = loadVocabulary(os.path.join(self.vocab_path, 'in_vocab'))
        self.slot_vocab = loadVocabulary(os.path.join(self.vocab_path, 'slot_vocab'))
        self.intent_vocab = loadVocabulary(os.path.join(self.vocab_path, 'intent_vocab'))

        # Create Training Model
        self.input_data = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        # Create Inference Model
        with tf.compat.v1.variable_scope('model'):
            inference_outputs = self.model.createModel(self.input_data, len(self.in_vocab['vocab']),
                                                       self.sequence_length, len(self.slot_vocab['vocab']),
                                                       len(self.intent_vocab['vocab']), layer_size=self.layer_size,
                                                       isTraining=False)

        inference_slot_output = tf.nn.softmax(inference_outputs[0], name='slot_output')
        inference_intent_output = tf.nn.softmax(inference_outputs[1], name='intent_output')

        self.inference_outputs = [inference_intent_output, inference_slot_output]

        self.saver = tf.compat.v1.train.Saver()

    def generate_dia_act(self, annot):
        if len(annot) > 0:
            log(["runner", "debug"], f'Sentence: {annot}')
            tmp_annot = annot.strip('.').strip('?').strip(',').strip('!')

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                self.saver.restore(sess, self.model_path)

                batch_in = []
                length = []
                in_data = []
                max_len = 0

                inp = sentenceToIds(tmp_annot, self.in_vocab)
                batch_in.append(np.array(inp))
                length.append(len(inp))

                if len(inp) > max_len:
                    max_len = len(inp)

                length = np.array(length)
                for i in batch_in:
                    in_data.append(padSentence(list(i), max_len, self.in_vocab))

                feed_dict= {self.input_data.name: in_data, self.sequence_length.name: length}
                ret = sess.run(self.inference_outputs, feed_dict)
                pred_intents = []
                for intent in ret[0]:
                    pred_intents.append(self.intent_vocab['rev'][np.argmax(intent)])

                log(["runner", "debug"], f"Intent: {pred_intents[0]}")

                pred_slots = []
                for slot in ret[1]:
                    pred = np.argmax(slot)
                    pred_slots.append(self.slot_vocab['rev'][pred])
                log(["runner", "debug"], f"IOB format: {pred_slots}")

                pred_tags = pred_slots + pred_intents
                diaact = self.parse_nlu_to_diaact(pred_tags, tmp_annot)
                log(["runner", "debug"], f"Diaact: {diaact}")
                return diaact
        else:
            return None

    def parse_nlu_to_diaact(self, nlu_vector, string):
        """ Parse BIO and Intent into Dia-Act """

        tmp = string + " EOS"
        words = tmp.lower().split(' ')

        diaact = {}
        diaact[const.INTENT] = const.INFORM
        diaact[const.REQUEST_SLOTS] = {}
        diaact[const.INFORM_SLOTS] = {}

        intent = nlu_vector[-1]
        index = 1
        pre_tag = nlu_vector[0]
        cur_tag = pre_tag
        pre_tag_index = 0

        slot_val_dict = {}

        while index<(len(nlu_vector)-1): # except last Intent tag
            cur_tag = nlu_vector[index]
            if cur_tag == 'O' and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('I-'):
                if cur_tag.split('-')[1] != pre_tag.split('-')[1]:
                    slot = pre_tag.split('-')[1]
                    slot_val_str = ' '.join(words[pre_tag_index:index])
                    slot_val_dict[slot] = slot_val_str
            elif cur_tag == 'O' and pre_tag.startswith('I-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str

            if cur_tag.startswith('B-'): pre_tag_index = index

            pre_tag = cur_tag
            index += 1

        if cur_tag.startswith('B-') or cur_tag.startswith('I-'):
            slot = cur_tag.split('-')[1]
            slot_val_str = ' '.join(words[pre_tag_index:-1])
            slot_val_dict[slot] = slot_val_str

        if intent != 'null':
            arr = intent.split('+')
            diaact[const.INTENT] = arr[0]
            diaact[const.REQUEST_SLOTS] = {}
            for ele in arr[1:]:
                #request_slots.append(ele)
                diaact[const.REQUEST_SLOTS][ele] = 'UNK'

        diaact[const.INFORM_SLOTS] = slot_val_dict

        # add rule here
        for slot in diaact[const.INFORM_SLOTS].keys():
            slot_val = diaact[const.INFORM_SLOTS][slot]
            if slot_val.startswith('bos'):
                slot_val = slot_val.replace('bos', '', 1)
                diaact[const.INFORM_SLOTS][slot] = slot_val.strip(' ')

        self.refine_diaact_by_rules(diaact)
        return diaact

    def refine_diaact_by_rules(self, diaact):
        """ refine the dia_act by rules """

        # rule for taskcomplete
        if const.REQUEST_SLOTS in diaact.keys():
            if const.TASK_COMPLETE_SLOT in diaact[const.REQUEST_SLOTS].keys():
                del diaact[const.REQUEST_SLOTS][const.TASK_COMPLETE_SLOT]
                diaact[const.INFORM_SLOTS][const.TASK_COMPLETE_SLOT] = const.PLACEHOLDER

            # rule for request
            if len(diaact[const.REQUEST_SLOTS])>0: diaact[const.INTENT] = const.REQUEST

    def diaact_penny_string(self, dia_act):
        """ Convert the Dia-Act into penny string """

        penny_str = ""
        penny_str = dia_act[const.INTENT] + "("
        for slot in dia_act[const.REQUEST_SLOTS].keys():
            penny_str += slot + ";"

        for slot in dia_act[const.INFORM_SLOTS].keys():
            slot_val_str = slot + "="
            if len(dia_act[const.INFORM_SLOTS][slot]) == 1:
                slot_val_str += dia_act[const.INFORM_SLOTS][slot][0]
            else:
                slot_val_str += "{"
                for slot_val in dia_act[const.INFORM_SLOTS][slot]:
                    slot_val_str += slot_val + "#"
                slot_val_str = slot_val_str[:-1]
                slot_val_str += "}"
            penny_str += slot_val_str + ";"

        if penny_str[-1] == ";": penny_str = penny_str[:-1]
        penny_str += ")"
        return penny_str
