#! /usr/bin/env python

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import os, time, datetime, copy
import utils.data_helpers as data_helpers
import gensim
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

# Data loading params
#tf.flags.DEFINE_string("train_data_file", "../data/reviews/review_16_laptop.train", "Data source for the positive data.")
#tf.flags.DEFINE_string("test_data_file", "../data/reviews/review_16_laptop.test", "Data source for the positive data.")
tf.flags.DEFINE_string("train_data_file", "../../data/reviews/review_16_restaurant.train", "Data source for the positive data.")
tf.flags.DEFINE_string("test_data_file", "../../data/reviews/review_16_restaurant.test", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print ('Loading Data...')
x_train_text, y_train, x_dev_text, y_dev, labels = data_helpers.load_data_and_labels_multi_class(FLAGS.train_data_file, FLAGS.test_data_file)

all_text = copy.deepcopy(x_train_text)
all_text.extend(x_dev_text)
max_document_length = max([len(x.split(" ")) for x in all_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(all_text)

x_train = np.array(list(vocab_processor.transform(x_train_text)))
x_dev = np.array(list(vocab_processor.transform(x_dev_text)))
y_train = np.array(y_train)
y_dev = np.array(y_dev)

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

for label_idx, label in enumerate(labels):

    y_train_single = []; y_dev_single = []
    for y in y_train[:,label_idx]:
        y_train_single.append([0,1] if y==1 else [1,0])
    for y in y_dev[:,label_idx]:
        y_dev_single.append([0,1] if y==1 else [1,0])
    y_train_single = np.array(y_train_single); y_dev_single = np.array(y_dev_single)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn=TextCNN(sequence_length = x_train.shape[1],
                    num_classes = y_train_single.shape[1],
                    vocab_size = len(vocab_processor.vocabulary_),
                    embedding_size = FLAGS.embedding_dim,
                    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda = FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_info", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
            prec_summary = tf.scalar_summary("precision", cnn.precision)
            recl_summary = tf.scalar_summary("recall", cnn.recall)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, prec_summary, recl_summary, grad_summaries_merged])
            train_summary_dir = out_dir #os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            config_pro = projector.ProjectorConfig()
            embedding = config_pro.embeddings.add()
            embedding.tensor_name = cnn.embedding.name
            embedding.metadata_path = os.path.join(out_dir, 'vocab_raw')
            projector.visualize_embeddings(train_summary_writer, config_pro)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = out_dir #os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver()#tf.global_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            vks = vocab_processor.vocabulary_._reverse_mapping
            with open(out_dir + '/vocab_raw', 'w+') as fout:
                for v in vks:
                    fout.write(v+'\n')

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            ## Initialize word_embedding
            #w2v_model = gensim.models.Word2Vec.load_word2vec_format('~/workspace/nlp/word2vec/models/GoogleNews-vectors-negative300.bin', binary=True)
            w2v_model = gensim.models.Word2Vec.load_word2vec_format('~/workspace/nlp/word2vec/models/vectors-reviews-electronics.bin', binary=True)
            #w2v_model = gensim.models.Word2Vec.load_word2vec_format('~/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin', binary=True)
            W_init = []
            for v in vks:
                try:
                    v_vec = w2v_model[v]
                except:
                    v_vec = np.random.uniform(-1, 1, 300)
                W_init.append(v_vec)
            W_init = np.array(W_init)
            sess.run(cnn.embedding.assign(W_init))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy, precision, recall= sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 100 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, recl {:g}".format(time_str, step, loss, accuracy, precision, recall))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, precision, recall = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, recl {:g}".format(time_str, step, loss, accuracy, precision, recall))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train_single)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev_single)#, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step % 10000 == 0:
                    break
