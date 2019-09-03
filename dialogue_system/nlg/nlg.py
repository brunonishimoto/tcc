'''
Created on Oct 17, 2016

--dia_act_nl_pairs.v6.json: agt and usr have their own NL.


@author: xiul
'''

import pickle
import copy, argparse, json
import numpy as np

import dialogue_system.constants as const
from .lstm_decoder_tanh import lstm_decoder_tanh


class NLG:
    def __init__(self, config):
        self.params = config['nlg']

        self.beam_size = self.params['beam_size']
        self.load_model(self.params['load_weights_file_path'])
        self.load_predefine_act_nl_pairs(self.params['load_predefined_file_path'])

    def post_process(self, pred_template, slot_val_dict, slot_dict):
        """ post_process to fill the slot in the template sentence """

        sentence = pred_template
        suffix = "_PLACEHOLDER"

        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = slot + suffix

            if slot == 'result' or slot == 'numberofpeople':
                continue

            if slot_vals == const.NO_MATCH:
                continue

            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence

        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = 'numberofpeople' + suffix
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence

        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence

        return sentence


    def convert_diaact_to_nl(self, dia_act, turn_msg):
        """ Convert Dia_Act into NL: Rule + Model """

        sentence = ""
        boolean_in = False

        if dia_act[const.INTENT] == const.MATCH_FOUND:
            dia_act[const.INTENT] = const.INFORM
            dia_act[const.INFORM_SLOTS].update({const.TASK_COMPLETE_SLOT: dia_act[const.INFORM_SLOTS]['ticket']})

        # remove I do not care slot in task(complete)
        if dia_act[const.INTENT] == const.INFORM and const.TASK_COMPLETE_SLOT in dia_act[const.INFORM_SLOTS].keys() and dia_act[const.INFORM_SLOTS][const.TASK_COMPLETE_SLOT] != const.NO_MATCH:
            inform_slot_set = dia_act[const.INFORM_SLOTS].keys()
            for slot in inform_slot_set:
                if dia_act[const.INFORM_SLOTS][slot] == const.I_DO_NOT_CARE: del dia_act[const.INFORM_SLOTS][slot]

        if dia_act[const.INTENT] in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][dia_act[const.INTENT]]:
                if set(ele[const.INFORM_SLOTS]) == set(dia_act[const.INFORM_SLOTS].keys()) and set(ele[const.REQUEST_SLOTS]) == set(dia_act[const.REQUEST_SLOTS].keys()):
                    sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
                    boolean_in = True
                    break

        if dia_act[const.INTENT] == const.INFORM and const.TASK_COMPLETE_SLOT in dia_act[const.INFORM_SLOTS].keys() and dia_act[const.INFORM_SLOTS][const.TASK_COMPLETE_SLOT] == const.NO_MATCH:
            sentence = "Oh sorry, there is no ticket available."

        if boolean_in == False: sentence = self.translate_diaact(dia_act)
        return sentence


    def translate_diaact(self, dia_act):
        """ prepare the diaact into vector representation, and generate the sentence by Model """

        word_dict = self.word_dict
        template_word_dict = self.template_word_dict
        act_dict = self.act_dict
        slot_dict = self.slot_dict
        inverse_word_dict = self.inverse_word_dict

        act_rep = np.zeros((1, len(act_dict)))
        act_rep[0, act_dict[dia_act[const.INTENT]]] = 1.0

        slot_rep_bit = 2
        slot_rep = np.zeros((1, len(slot_dict)*slot_rep_bit))

        suffix = "_PLACEHOLDER"
        if self.model_params['dia_slot_val'] == 2 or self.model_params['dia_slot_val'] == 3:
            word_rep = np.zeros((1, len(template_word_dict)))
            words = np.zeros((1, len(template_word_dict)))
            words[0, template_word_dict['s_o_s']] = 1.0
        else:
            word_rep = np.zeros((1, len(word_dict)))
            words = np.zeros((1, len(word_dict)))
            words[0, word_dict['s_o_s']] = 1.0

        for slot in dia_act[const.INFORM_SLOTS].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit] = 1.0

            for slot_val in dia_act[const.INFORM_SLOTS][slot]:
                if self.model_params['dia_slot_val'] == 2:
                    slot_placeholder = slot + suffix
                    if slot_placeholder in template_word_dict.keys():
                        word_rep[0, template_word_dict[slot_placeholder]] = 1.0
                elif self.model_params['dia_slot_val'] == 1:
                    if slot_val in word_dict.keys():
                        word_rep[0, word_dict[slot_val]] = 1.0

        for slot in dia_act[const.REQUEST_SLOTS].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit + 1] = 1.0

        if self.model_params['dia_slot_val'] == 0 or self.model_params['dia_slot_val'] == 3:
            final_representation = np.hstack([act_rep, slot_rep])
        else: # dia_slot_val = 1, 2
            final_representation = np.hstack([act_rep, slot_rep, word_rep])

        dia_act_rep = {}
        dia_act_rep[const.INTENT] = final_representation
        dia_act_rep['words'] = words

        #pred_ys, pred_words = nlg_model['model'].forward(inverse_word_dict, dia_act_rep, nlg_model['params'], predict_model=True)
        pred_ys, pred_words = self.model.beam_forward(inverse_word_dict, dia_act_rep, self.params, predict_model=True)
        pred_sentence = ' '.join(pred_words[:-1])
        sentence = self.post_process(pred_sentence, dia_act[const.INFORM_SLOTS], slot_dict)

        return sentence


    def load_model(self, model_path):
        """ load the trained NLG model """

        model_params = pickle.load(open(model_path, 'rb'), encoding='latin1')

        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]

        if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
            diaact_input_size = model_params['model']['Wah'].shape[0]
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm_decoder_tanh(diaact_input_size, input_size, hidden_size, output_size)

        rnnmodel.model = copy.deepcopy(model_params['model'])
        model_params['params']['beam_size'] = self.beam_size

        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.template_word_dict = copy.deepcopy(model_params['template_word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.inverse_word_dict = {self.template_word_dict[k]:k for k in self.template_word_dict.keys()}
        self.model_params = copy.deepcopy(model_params['params'])


    def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
        """ Replace the slots with its values """

        sentence = template_sentence
        counter = 0
        for slot in dia_act[const.INFORM_SLOTS].keys():
            slot_val = dia_act[const.INFORM_SLOTS][slot]
            if slot_val == const.NO_MATCH:
                sentence = slot + " is not available!"
                break
            elif slot_val == const.I_DO_NOT_CARE:
                counter += 1
                sentence = sentence.replace(f'${slot}$', '', 1)
                continues

            sentence = sentence.replace(f'${slot}$', slot_val, 1)

        if counter > 0 and counter == len(dia_act[const.INFORM_SLOTS]):
            sentence = const.I_DO_NOT_CARE

        return sentence


    def load_predefine_act_nl_pairs(self, path):
        """ Load some pre-defined Dia_Act&NL Pairs from file """

        self.diaact_nl_pairs = json.load(open(path, 'rb'))

        # for key in self.diaact_nl_pairs['dia_acts'].keys():
        #     for ele in self.diaact_nl_pairs['dia_acts'][key]:
        #         ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
        #         ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue


def main(params):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print ("User Simulator Parameters:")
    print (json.dumps(params, indent=2))

    main(params)
