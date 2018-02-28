import sys
sys.path.append("..")

import math
import inspect

from Data.VisualGenome.models import *
from FilesManager.FilesManager import FilesManager
from Module import Module
import tensorflow as tf
import numpy as np
import os
import cPickle
from random import shuffle

from Utils.Logger import Logger

# feature sizes
NOF_PREDICATES = 51
NOF_OBJECTS = 150

# save model every number of iterations
SAVE_MODEL_ITERATIONS = 5

# test every number of iterations
TEST_ITERATIONS = 1


def test(labels_predicate, labels_object, out_confidence_predicate_val, out_confidence_object_val):
    """
    returns a dictionary with statistics about object, predicate and relationship accuracy in this image
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_object: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_confidence_predicate_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_confidence_object_val: confidence of image objects - shape (N, NOF_OBJECTS)
    :return: see description
    """
    predicats_gt = np.argmax(labels_predicate, axis=2)
    objects_gt = np.argmax(labels_object, axis=1)
    predicats_pred = np.argmax(out_confidence_predicate_val, axis=2)
    predicats_pred_no_neg = np.argmax(out_confidence_predicate_val[:, :, :NOF_PREDICATES - 1], axis=2)
    objects_pred = np.argmax(out_confidence_object_val, axis=1)

    # noinspection PyDictCreation
    results = {}
    # number of objects
    results["obj_total"] = objects_gt.shape[0]
    # number of predicates / relationships
    results["predicates_total"] = predicats_gt.shape[0] * predicats_gt.shape[1]
    # number of positive predicates / relationships
    pos_indices = np.where(predicats_gt != NOF_PREDICATES - 1)
    results["predicates_pos_total"] = pos_indices[0].shape[0]

    # number of object correct predictions
    results["obj_correct"] = np.sum(objects_gt == objects_pred)
    # number of correct predicate
    results["predicates_correct"] = np.sum(predicats_gt == predicats_pred)
    # number of correct positive predicates
    predicates_gt_pos = predicats_gt[pos_indices]
    predicates_pred_pos = predicats_pred_no_neg[pos_indices]
    results["predicates_pos_correct"] = np.sum(predicates_gt_pos == predicates_pred_pos)
    # number of correct relationships
    object_true_indices = np.where(objects_gt == objects_pred)
    predicates_gt_true = predicats_gt[object_true_indices[0], :][:, object_true_indices[0]]
    predicates_pred_true = predicats_pred[object_true_indices[0], :][:, object_true_indices[0]]
    predicates_pred_true_pos = predicats_pred_no_neg[object_true_indices[0], :][:, object_true_indices[0]]
    results["relationships_correct"] = np.sum(predicates_gt_true == predicates_pred_true)
    # number of correct positive relationships
    pos_true_indices = np.where(predicates_gt_true != NOF_PREDICATES - 1)
    predicates_gt_pos_true = predicates_gt_true[pos_true_indices]
    predicates_pred_pos_true = predicates_pred_true_pos[pos_true_indices]
    results["relationships_pos_correct"] = np.sum(predicates_gt_pos_true == predicates_pred_pos_true)

    return results


def train(name="test",
          gpi_type="Linguistic",
          nof_iterations=100,
          learning_rate=0.0001,
          learning_rate_steps=1000,
          learning_rate_decay=0.5,
          load_module_name="module.ckpt",
          use_saved_module=False,
          rnn_steps=1,
          batch_size=200,
          pred_pos_neg_ratio=10,
          lr_object_coeff=1,
          including_object=False,
          layers=[],
          reg_factor=0.0,
          gpu=0):
    """
    Train SGP module given train parameters and module hyper-parameters
    :param name: name of the train session
    :param nof_iterations: number of epochs
    :param learning_rate:
    :param learning_rate_steps: decay after number of steps
    :param learning_rate_decay: the factor to decay the learning rate
    :param load_module_name: name of already trained module weights to load
    :param use_saved_module: start from already train module
    :param rnn_steps: how many times to apply SGP
    :param batch_size: number of images in each mini-batch
    :param pred_pos_neg_ratio: Set the loss ratio between positive and negatives (not labeled) predicates
    :param lr_object_coeff: Set the loss ratio between objects and predicates
    :param including_object: Whether to  predict objects as well
    :param layers: list of sizes of the hidden layer of the predicate and object classifier
    :param reg_factor: L2 regulizer factor
    :param gpu: gpu number to use for the training
    :return: nothing
    """
    # get filesmanager
    filesmanager = FilesManager()

    # create logger
    logger_path = filesmanager.get_file_path("logs")
    logger_path = os.path.join(logger_path, name)
    logger = Logger(name, logger_path)

    # print train params
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    logger.log('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        logger.log("    %s = %s" % (i, values[i]))

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.log("os.environ[\"CUDA_VISIBLE_DEVICES\"] = " + str(gpu))

    # create module
    module = Module(gpi_type=gpi_type, nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                    is_train=True,
                    learning_rate=learning_rate, learning_rate_steps=learning_rate_steps,
                    learning_rate_decay=learning_rate_decay,
                    rnn_steps=rnn_steps,
                    lr_object_coeff=lr_object_coeff,
                    including_object=including_object,
                    layers=layers)

    ##
    # get module place holders
    #
    # get input place holders
    confidence_predicate_ph, confidence_object_ph = module.get_in_ph()
    # get labels place holders
    labels_predicate_ph, labels_object_ph, labels_coeff_loss_ph = module.get_labels_ph()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = module.get_module_loss()

    ##
    # get module output
    out_predicate_probes, out_object_probes = module.get_output()

    # Initialize the Computational Graph
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    variables = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore = variables
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        # Restore variables from disk.
        module_path = filesmanager.get_file_path("sg_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        if os.path.exists(module_path_load + ".index") and use_saved_module:
            saver.restore(sess, module_path_load)
            logger.log("Model restored.")
        else:
            sess.run(init)

        # get object labels to ids and predicate labels to ids
        # TBD _, object_ids, predicate_ids = get_filtered_data(filtered_data_file_name="mini_filtered_data", category='entities_visual_module')
        object_ids = []
        predicate_ids = []
        # train entities
        entities_path = filesmanager.get_file_path("data.visual_genome.detections_v4")

        train_files_list = ["Sat_Nov_11_20:43:42_2017/predicated_entities_0_to_1000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_1000_to_2000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_2000_to_3000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_3000_to_4000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_4000_to_5000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_5000_to_6000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_6000_to_7000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_7000_to_8000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_8000_to_9000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_9000_to_10000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_10000_to_11000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_11000_to_12000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_12000_to_13000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_13000_to_14000.p",
                            "Sat_Nov_11_20:43:42_2017/predicated_entities_14000_to_15000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_0_to_1000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_1000_to_2000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_2000_to_3000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_3000_to_4000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_4000_to_5000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_5000_to_6000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_6000_to_7000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_7000_to_8000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_8000_to_9000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_9000_to_10000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_10000_to_11000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_11000_to_12000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_12000_to_13000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_13000_to_14000.p",
                            "Sat_Nov_11_20:47:34_2017/predicated_entities_14000_to_15000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_0_to_1000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_1000_to_2000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_2000_to_3000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_3000_to_4000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_4000_to_5000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_5000_to_6000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_6000_to_7000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_7000_to_8000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_8000_to_9000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_9000_to_10000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_10000_to_11000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_11000_to_12000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_12000_to_13000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_13000_to_14000.p",
                            "Sat_Nov_11_20:48:52_2017/predicated_entities_14000_to_15000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_0_to_1000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_1000_to_2000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_2000_to_3000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_3000_to_4000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_4000_to_5000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_5000_to_6000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_6000_to_7000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_7000_to_8000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_8000_to_9000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_9000_to_10000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_10000_to_11000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_11000_to_12000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_12000_to_13000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_13000_to_14000.p",
                            "Sat_Nov_11_20:50:15_2017/predicated_entities_14000_to_15000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_0_to_1000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_1000_to_2000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_2000_to_3000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_3000_to_4000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_4000_to_5000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_5000_to_6000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_6000_to_7000.p",
                            "Sat_Nov_11_20:56:19_2017/predicated_entities_7000_to_8000.p"]
        train_files_list = ["Sat_Nov_11_20:43:42_2017/predicated_entities_0_to_1000.p"]
        validation_files_list = ["Sat_Nov_11_20:56:19_2017/predicated_entities_8000_to_9000.p",
                                 "Sat_Nov_11_20:56:19_2017/predicated_entities_9000_to_10000.p",
                                 "Sat_Nov_11_20:56:19_2017/predicated_entities_10000_to_11000.p",
                                 "Sat_Nov_11_20:56:19_2017/predicated_entities_11000_to_12000.p",
                                 "Sat_Nov_11_20:56:19_2017/predicated_entities_12000_to_13000.p"]

        test_files_list = ["Sat_Nov_11_21:36:12_2017/predicated_entities_0_to_1000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_1000_to_2000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_2000_to_3000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_3000_to_4000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_4000_to_5000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_5000_to_6000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_6000_to_7000.p",
                           "Sat_Nov_11_21:36:12_2017/predicated_entities_7000_to_7500.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_0_to_1000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_1000_to_2000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_2000_to_3000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_3000_to_4000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_4000_to_5000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_5000_to_6000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_6000_to_7000.p",
                           "Sat_Nov_11_21:38:29_2017/predicated_entities_7000_to_7500.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_0_to_1000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_1000_to_2000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_2000_to_3000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_3000_to_4000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_4000_to_5000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_5000_to_6000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_6000_to_7000.p",
                           "Sat_Nov_11_21:42:07_2017/predicated_entities_7000_to_7500.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_0_to_1000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_1000_to_2000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_2000_to_3000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_3000_to_4000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_4000_to_5000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_5000_to_6000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_6000_to_7000.p",
                           "Sat_Nov_11_21:43:18_2017/predicated_entities_7000_to_7500.p",
                           "Sat_Nov_11_21:59:10_2017/predicated_entities_0_to_1000.p",
                           "Sat_Nov_11_21:59:10_2017/predicated_entities_1000_to_2000.p",
                           "Sat_Nov_11_21:59:10_2017/predicated_entities_2000_to_2422.p"]

        # create one hot vector for predicate_negative (i.e. not labeled)
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1

        # object embedding
        embed_obj = FilesManager().load_file("language_module.word2vec.object_embeddings")
        embed_pred = FilesManager().load_file("language_module.word2vec.predicate_embeddings")
        embed_pred = np.concatenate((embed_pred, np.zeros(embed_pred[:1].shape)),
                                    axis=0)  # concat negative represntation

        # train module
        lr = learning_rate
        best_test_loss = -1
        for epoch in xrange(1, nof_iterations):
            accum_results = None
            total_loss = 0
            steps = []
            # shuffle entities groups
            shuffle(train_files_list)
            # read data
            file_index = -1
            for file_name in train_files_list:

                file_index += 1

                # load data from file
                file_path = os.path.join(entities_path, file_name)
                file_handle = open(file_path, "rb")
                train_entities = cPickle.load(file_handle)
                file_handle.close()
                shuffle(train_entities)

                for entity in train_entities:

                    # set diagonal to be negative predicate (no relation for a single object)
                    indices = np.arange(entity.predicates_probes.shape[0])
                    entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
                    entity.predicates_labels[indices, indices, :] = predicate_neg
                    entity.predicates_probes[indices, indices, :] = predicate_neg

                    # spatial features
                    obj_bb = np.zeros((len(entity.objects), 4))
                    for obj_id in range(len(entity.objects)):
                        obj_bb[obj_id][0] = entity.objects[obj_id].x / 1200.0
                        obj_bb[obj_id][1] = entity.objects[obj_id].y / 1200.0
                        obj_bb[obj_id][2] = (entity.objects[obj_id].x + entity.objects[obj_id].width) / 1200.0
                        obj_bb[obj_id][3] = (entity.objects[obj_id].y + entity.objects[obj_id].height) / 1200.0

                    # filter non mixed cases
                    predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
                    if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                            predicates_neg_labels) == 0:
                        continue

                    if including_object:
                        in_object_confidence = entity.objects_outputs_with_no_activations
                    else:
                        in_object_confidence = entity.objects_labels * 1000

                    # give lower weight to negatives
                    coeff_factor = np.ones(predicates_neg_labels.shape)
                    factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                        predicates_neg_labels) / pred_pos_neg_ratio
                    coeff_factor[predicates_neg_labels == 1] *= factor

                    coeff_factor[indices, indices] = 0

                    # create the feed dictionary
                    feed_dict = {confidence_predicate_ph: entity.predicates_outputs_with_no_activation,
                                 confidence_object_ph: in_object_confidence,
                                 module.entity_bb_ph : obj_bb,
                                 module.word_embed_entities_ph: embed_obj, module.word_embed_relations_ph: embed_pred,
                                 labels_predicate_ph: entity.predicates_labels, labels_object_ph: entity.objects_labels,
                                 labels_coeff_loss_ph: coeff_factor.reshape((-1)), module.lr_ph: lr}

                    # run the network
                    out_predicate_probes_val, out_object_probes_val, loss_val, gradients_val = \
                        sess.run([out_predicate_probes, out_object_probes, loss, gradients],
                                 feed_dict=feed_dict)
                    if math.isnan(loss_val):
                        print("NAN")
                        continue

                    # set diagonal to be neg (in order not to take into account in statistics)
                    out_predicate_probes_val[indices, indices, :] = predicate_neg

                    # append gradient to list (will be applied as a batch of entities)
                    steps.append(gradients_val)

                    # statistic
                    total_loss += loss_val

                    # unmask for sanity check (getting results of feature extractor)
                    # out_predicate_probes_val = entity.predicates_probes
                    # out_object_probes_val = entity.objects_probs
                    results = test(entity.predicates_labels, entity.objects_labels, out_predicate_probes_val,
                                   out_object_probes_val)

                    # accumulate results
                    if accum_results is None:
                        accum_results = results
                    else:
                        for key in results:
                            accum_results[key] += results[key]

                    if len(steps) == batch_size:
                        # apply steps
                        step = steps[0]
                        feed_grad_apply_dict = {grad_placeholder[j][0]: step[j][0] for j in
                                                xrange(len(grad_placeholder))}
                        for i in xrange(1, len(steps)):
                            step = steps[i]
                            for j in xrange(len(grad_placeholder)):
                                feed_grad_apply_dict[grad_placeholder[j][0]] += step[j][0]

                        feed_grad_apply_dict[module.lr_ph] = lr
                        sess.run([train_step], feed_dict=feed_grad_apply_dict)
                        steps = []
                # print stat - per file just for the first epoch
                if epoch == 1:
                    obj_accuracy = float(accum_results['obj_correct']) / accum_results['obj_total']
                    predicate_pos_accuracy = float(accum_results['predicates_pos_correct']) / accum_results[
                        'predicates_pos_total']
                    relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                        'predicates_pos_total']
                    logger.log("iter %d.%d - obj %f - pred %f - relation %f" %
                           (epoch, file_index, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy))

            # print stat
            obj_accuracy = float(accum_results['obj_correct']) / accum_results['obj_total']
            predicate_pos_accuracy = float(accum_results['predicates_pos_correct']) / accum_results[
                'predicates_pos_total']
            predicate_all_accuracy = float(accum_results['predicates_correct']) / accum_results['predicates_total']
            relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                'predicates_pos_total']
            relationships_all_accuracy = float(accum_results['relationships_correct']) / accum_results[
                'predicates_total']

            logger.log("iter %d - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - lr %f" %
                       (epoch, total_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                        predicate_all_accuracy, relationships_all_accuracy, lr))

            if epoch % TEST_ITERATIONS == 0:
                total_test_loss = 0
                accum_test_results = None
                correct_predicate = 0
                total_predicate = 0

                for file_name in validation_files_list:

                    # load data from file
                    file_path = os.path.join(entities_path, file_name)
                    file_handle = open(file_path, "rb")
                    validation_entities = cPickle.load(file_handle)
                    file_handle.close()

                    for entity in validation_entities:

                        # set diagonal to be neg
                        indices = np.arange(entity.predicates_probes.shape[0])
                        entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
                        entity.predicates_labels[indices, indices, :] = predicate_neg
                        entity.predicates_probes[indices, indices, :] = predicate_neg

                        # get shape of extended object to be used by the module
                        extended_confidence_object_shape = np.asarray(entity.predicates_probes.shape)
                        extended_confidence_object_shape[2] = NOF_OBJECTS

                        # spatial features
                        obj_bb = np.zeros((len(entity.objects), 4))
                        for obj_id in range(len(entity.objects)):
                            obj_bb[obj_id][0] = entity.objects[obj_id].x / 1200.0
                            obj_bb[obj_id][1] = entity.objects[obj_id].y / 1200.0
                            obj_bb[obj_id][2] = (entity.objects[obj_id].x + entity.objects[obj_id].width) / 1200.0
                            obj_bb[obj_id][3] = (entity.objects[obj_id].y + entity.objects[obj_id].height) / 1200.0

                        # filter non mixed cases
                        predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
                        if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                                predicates_neg_labels) == 0:
                            continue

                        # give lower weight to negatives
                        coeff_factor = np.ones(predicates_neg_labels.shape)
                        factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                            predicates_neg_labels) / pred_pos_neg_ratio
                        coeff_factor[predicates_neg_labels == 1] *= factor
                        coeff_factor[indices, indices] = 0
                        coeff_factor[predicates_neg_labels == 1] = 0

                        if including_object:
                            in_object_confidence = entity.objects_outputs_with_no_activations
                        else:
                            in_object_confidence = entity.objects_labels * 1000

                        # create the feed dictionary
                        feed_dict = {confidence_predicate_ph: entity.predicates_outputs_with_no_activation,
                                     confidence_object_ph: in_object_confidence,
                                     module.entity_bb_ph: obj_bb,
                                     module.word_embed_entities_ph: embed_obj,
                                     module.phase_ph: False,
                                     labels_predicate_ph: entity.predicates_labels,
                                     labels_object_ph: entity.objects_labels,
                                     labels_coeff_loss_ph: coeff_factor.reshape((-1))}

                        # run the network
                        out_predicate_probes_val, out_object_probes_val, loss_val = sess.run(
                            [out_predicate_probes, out_object_probes, loss],
                            feed_dict=feed_dict)

                        # set diagonal to be neg (in order not to take into account in statistics)
                        out_predicate_probes_val[indices, indices, :] = predicate_neg

                        # statistic
                        total_test_loss += loss_val

                        # statistics
                        # uncomment for sanity check (to get features extractor statistics)
                        # out_predicate_probes_val =entity.predicates_probes
                        # out_object_probes_val = entity.objects_probs
                        results = test(entity.predicates_labels, entity.objects_labels,
                                       out_predicate_probes_val, out_object_probes_val)

                        # accumulate results
                        if accum_test_results is None:
                            accum_test_results = results
                        else:
                            for key in results:
                                accum_test_results[key] += results[key]

                        # eval per predicate
                        correct_predicate_image, total_predicate_image = predicate_class_recall(
                            entity.predicates_labels,
                            out_predicate_probes_val)
                        correct_predicate += np.sum(correct_predicate_image[:NOF_PREDICATES - 2])
                        total_predicate += np.sum(total_predicate_image[:NOF_PREDICATES - 2])

                # print stat
                obj_accuracy = float(accum_test_results['obj_correct']) / accum_test_results['obj_total']
                predicate_pos_accuracy = float(accum_test_results['predicates_pos_correct']) / accum_test_results[
                    'predicates_pos_total']
                predicate_all_accuracy = float(accum_test_results['predicates_correct']) / accum_test_results[
                    'predicates_total']
                relationships_pos_accuracy = float(accum_test_results['relationships_pos_correct']) / \
                                             accum_test_results[
                                                 'predicates_pos_total']
                relationships_all_accuracy = float(accum_test_results['relationships_correct']) / accum_test_results[
                    'predicates_total']

                logger.log("TEST - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - top5 %f" %
                           (total_test_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                            predicate_all_accuracy, relationships_all_accuracy,
                            float(correct_predicate) / total_predicate))

                # save best module so far
                if best_test_loss == -1 or total_test_loss < best_test_loss:
                    module_path_save = os.path.join(module_path, name + "_best_module.ckpt")
                    save_path = saver.save(sess, module_path_save)
                    logger.log("Model saved in file: %s" % save_path)
                    best_test_loss = total_test_loss

            # save module
            if epoch % SAVE_MODEL_ITERATIONS == 0:
                module_path_save = os.path.join(module_path, name + "_module.ckpt")
                save_path = saver.save(sess, module_path_save)
                logger.log("Model saved in file: %s" % save_path)

            # learning rate decay
            if (epoch % learning_rate_steps) == 0:
                lr *= learning_rate_decay


def predicate_class_recall(labels_predicate, out_confidence_predicate_val, k=5):
    """
    Predicate Classification - Examine the model performance on predicates classification in isolation from other factors
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param out_confidence_predicate_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param k: k most confident predictions to consider
    :return: correct vector (number of times predicate gt appears in top k most confident predicates),
             total vector ( number of gts per predicate)
    """
    correct = np.zeros(NOF_PREDICATES)
    total = np.zeros(NOF_PREDICATES)

    # one hot vector to actual gt labels
    predicates_gt = np.argmax(labels_predicate, axis=2)

    # number of objects in the image
    N = out_confidence_predicate_val.shape[0]

    # run over each prediction
    for subject_index in range(N):
        for object_index in range(N):
            # get predicate class
            predicate_class = predicates_gt[subject_index][object_index]
            # get predicate probabilities
            predicate_prob = out_confidence_predicate_val[subject_index][object_index]

            max_k_predictions = np.argsort(predicate_prob)[-k:]
            found = np.where(predicate_class == max_k_predictions)[0]
            if len(found) != 0:
                correct[predicate_class] += 1
            total[predicate_class] += 1

    return correct, total


if __name__ == "__main__":
    filemanager = FilesManager()

    params = filemanager.load_file("sg_module.train.params")

    name = params["name"]
    gpi_type = params["gpi_type"]
    learning_rate = params["learning_rate"]
    learning_rate_steps = params["learning_rate_steps"]
    learning_rate_decay = params["learning_rate_decay"]
    nof_iterations = params["nof_iterations"]
    load_model_name = params["load_model_name"]
    use_saved_model = params["use_saved_model"]
    rnn_steps = params["rnn_steps"]
    batch_size = params["batch_size"]
    predicate_pos_neg_ratio = params["predicate_pos_neg_ratio"]
    lr_object_coeff = params["lr_object_coeff"]
    including_object = params["including_object"]
    layers = params["layers"]
    reg_factor = params["reg_factor"]
    gpu = params["gpu"]

    train(name, gpi_type, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
          use_saved_model, rnn_steps, batch_size, predicate_pos_neg_ratio, lr_object_coeff, including_object, layers,
          reg_factor, gpu)
