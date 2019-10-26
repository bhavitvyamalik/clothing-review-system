from bert import run_classifier
import os
import tensorflow as tf
import json
import operator

PATH="C:\\Users\\bhavitvyamalik\\Desktop\\Training/task 1\\uncased_L-12_H-768_A-12"
tokenization = run_classifier.tokenization
init_checkpoint = os.path.join(PATH, 'model.ckpt')
bert_config_file = os.path.join(PATH, 'bert_config.json')
vocab_file = os.path.join(PATH, 'vocab.txt')
processor = run_classifier.ColaProcessor()
label_list = processor.get_labels()


BATCH_SIZE = 8
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 500
OUTPUT_DIR = "./bert_output/output"

#variables that needed to be modified
labels = ["0", "1", "2", "3", "4", "5"] #modify based on the labels that you have
label_list=labels
MAX_SEQ_LENGTH = 128
is_lower_case = True
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8

tpu_cluster_resolver = None
tokenization.validate_case_matches_checkpoint(is_lower_case, init_checkpoint)


class Reviews:

    def __init__(self):

        self.bert_config = run_classifier.modeling.BertConfig.from_json_file(bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=is_lower_case)
        self.is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=OUTPUT_DIR,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=ITERATIONS_PER_LOOP,
                num_shards=NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        self.model_fn = run_classifier.model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(label_list),
            init_checkpoint=init_checkpoint,
            learning_rate=5e-5,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=False,
            use_one_hot_embeddings=False)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            predict_batch_size=BATCH_SIZE)


    def getListPrediction(self, in_sentences):
        #print(in_sentences)
        #1
        input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = "0") for x in in_sentences] # here, "" is just a dummy label

        #2
        input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, self.tokenizer)

        #3
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

        #4
        predictions = self.estimator.predict(input_fn=predict_input_fn)

        return predictions
