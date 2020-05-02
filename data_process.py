import json
from collections import namedtuple
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import config
import utils
from batching import pad_batch_data


class TriggerSequenceLabelReader():
    """TriggerSequenceLabelReader
    """
    def __init__(self,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)

        self.max_seq_len = max_seq_len

        labels_map = {}  # label
        for line in utils.read_by_lines(config.trigger_label_map):
            arr = line.split("\t")
            labels_map[arr[0]] = int(arr[1])
        # self.tokenizer = tokenization.FullTokenizer(
        #     vocab_file=vocab_path, do_lower_case=do_lower_case)
        # self.vocab = self.tokenizer.vocab
        # self.pad_id = self.vocab["[PAD]"]
        # self.cls_id = self.vocab["[CLS]"]
        # self.sep_id = self.vocab["[SEP]"]
        # self.in_tokens = in_tokens
        # self.is_inference = is_inference
        # self.for_cn = for_cn
        # self.task_id = task_id

        # np.random.seed(random_seed)

        # self.is_classify = is_classify
        # self.is_regression = is_regression
        # self.current_example = 0
        # self.current_epoch = 0
        # self.num_examples = 0

        self.label_map = labels_map

    def _process_examples_by_json(self, input_data):
        """_examples_by_json"""
        def process_sent_ori_2_new(sent, start, end):
            """process_sent_ori_2_new"""
            words = list(sent)
            sent_ori_2_new_index = {}
            new_words = []
            new_start, new_end = -1, -1
            for i, w in enumerate(words):
                if i == start:
                    new_start = len(new_words)
                if i == end:
                    new_end = len(new_words)
                if len(w.strip()) == 0:
                    sent_ori_2_new_index[i] = -1
                    if i == end:
                        new_end -= 1
                    if i == start:
                        start += 1
                else:
                    sent_ori_2_new_index[i] = len(new_words)
                    new_words.append(w)
            if new_end == len(new_words):
                new_end = len(new_words) - 1

            return [words, new_words, sent_ori_2_new_index, new_start, new_end]

        examples = []
        k = 0
        Example = namedtuple('Example', [
            "id", "text_a", "label", "ori_text", "ori_2_new_index", "sentence"
        ])
        for data in input_data:
            event_id = data["event_id"]
            sentence = data["text"]
            trigger_start = data["trigger_start_index"]
            trigger_text = data["trigger"]
            trigger_end = trigger_start + len(trigger_text) - 1
            event_type = data["event_type"]
            (sent_words, new_sent_words, ori_2_new_sent_index,
             new_trigger_start,
             new_trigger_end) = process_sent_ori_2_new(sentence.lower(),
                                                       trigger_start,
                                                       trigger_end)
            new_sent_labels = [u"O"] * len(new_sent_words)
            for i in range(new_trigger_start, new_trigger_end + 1):
                if i == new_trigger_start:
                    new_sent_labels[i] = u"B-{}".format(event_type)
                else:
                    new_sent_labels[i] = u"I-{}".format(event_type)
            example = Example(id=event_id,
                              text_a=u" ".join(new_sent_words),
                              label=u" ".join(new_sent_labels),
                              ori_text=sent_words,
                              ori_2_new_index=ori_2_new_sent_index,
                              sentence=sentence)

            if k > 0:
                print(u"example {} : {}".format(
                    k, json.dumps(example._asdict(), ensure_ascii=False)))
            k -= 1
            examples.append(example)
        return examples

    def _read_json_file(self, input_file):
        """_read_json_file"""
        input_data = []
        with open(input_file, "r", encoding='utf8') as f:
            for line in f:
                d_json = json.loads(line.strip())
                input_data.append(d_json)
        examples = self._process_examples_by_json(input_data)
        return examples

    def get_examples_by_file(self, input_file):
        """get_examples_by_file"""
        return self._read_json_file(input_file)

    def _pad_batch_records(self, batch_records):
        """_pad_batch_records"""
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [
            record.text_type_ids for record in batch_records
        ]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(batch_text_type_ids,
                                              pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(batch_position_ids,
                                             pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(batch_label_ids,
                                          pad_idx=len(self.label_map) - 1)
        padded_task_ids = np.ones_like(padded_token_ids,
                                       dtype="int64") * self.task_id

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        """_reseg_token_label"""
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels


    def whitespace_tokenize(text):
        """Runs basic whitespace cleaning and splitting on a peice of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """_convert_example_to_record"""
        tokens = self.whitespace_tokenize(example.text_a)
        labels = self.whitespace_tokenize(example.label)
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id
                     ] + [self.label_map[label]
                          for label in labels] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
        record = Record(token_ids=token_ids,
                        text_type_ids=text_type_ids,
                        position_ids=position_ids,
                        label_ids=label_ids)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        k = 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
            if k > 0:
                print(u"feature {} : {}".format(
                    k, json.dumps(record._asdict(), ensure_ascii=False)))
            k -= 1
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        """get_num_examples"""
        examples = self._read_json_file(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        """data_generator"""
        examples = self._read_json_file(input_file)

        def wrapper():
            """wrapper"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(examples,
                                                           batch_size,
                                                           phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        return wrapper

reader = TriggerSequenceLabelReader()

# *****
#  Examples
#  ("id", "text_a", "label", "ori_text", "ori_2_new_index", "sentence") 
#  id = 'cba11b5059495e635b4f95e7484b2684_裁员'
#  text_a = '消 失 的 “ 外 企 光 环 ” ， 5 月 份 在 华 裁 员 9 0 0 余 人 ， 香 饽 饽 变 “ 臭 ” 了'
#  label = 'O O O O O O O O O O O O O O O B-组织关系-裁员 I-组织关系-裁员 O O O O O O O O O O O O O O'
#  ori_text = ['消', '失', '的', '“', '外', '企', '光', '环', '”', '，', '5', '月', '份', '在', '华', '裁', '员', '9', '0', '0', '余', '人', '，', '香', '饽', '饽', '变', '“', '臭', '”', '了']
#  ori_2_new_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30}
#  sentence = '消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了')
# 
# *****
# examples = reader._read_json_file("./data/dev.json")
# record = reader._convert_example_to_record(examples[0])
# print(record)
# print(examples[:3])
data = reader.data_generator("./dev.json", 8, 1)
for it in data:
    print(it[0])
    break