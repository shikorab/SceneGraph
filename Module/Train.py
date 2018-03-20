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


def test(labels_relation, labels_entity, out_confidence_relation_val, out_confidence_entity_val):
    """
    returns a dictionary with statistics about object, predicate and relationship accuracy in this image
    :param labels_relation: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_entity: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_confidence_relation_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_confidence_entity_val: confidence of image objects - shape (N, NOF_OBJECTS)
    :return: see description
    """
    relation_gt = np.argmax(labels_relation, axis=2)
    entity_gt = np.argmax(labels_entity, axis=1)
    relation_pred = np.argmax(out_confidence_relation_val, axis=2)
    relations_pred_no_neg = np.argmax(out_confidence_relation_val[:, :, :NOF_PREDICATES - 1], axis=2)
    entities_pred = np.argmax(out_confidence_entity_val, axis=1)

    # noinspection PyDictCreation
    results = {}
    # number of objects
    results["entity_total"] = entity_gt.shape[0]
    # number of predicates / relationships
    results["relations_total"] = relation_gt.shape[0] * relation_gt.shape[1]
    # number of positive predicates / relationships
    pos_indices = np.where(relation_gt != NOF_PREDICATES - 1)
    results["relations_pos_total"] = pos_indices[0].shape[0]

    # number of object correct predictions
    results["entity_correct"] = np.sum(entity_gt == entities_pred)
    # number of correct predicate
    results["relations_correct"] = np.sum(relation_gt == relation_pred)
    # number of correct positive predicates
    relations_gt_pos = relation_gt[pos_indices]
    relations_pred_pos = relations_pred_no_neg[pos_indices]
    results["relations_pos_correct"] = np.sum(relations_gt_pos == relations_pred_pos)
    # number of correct relationships
    entity_true_indices = np.where(entity_gt == entities_pred)
    relations_gt_true = relation_gt[entity_true_indices[0], :][:, entity_true_indices[0]]
    relations_pred_true = relation_pred[entity_true_indices[0], :][:, entity_true_indices[0]]
    relations_pred_true_pos = relations_pred_no_neg[entity_true_indices[0], :][:, entity_true_indices[0]]
    results["relationships_correct"] = np.sum(relations_gt_true == relations_pred_true)
    # number of correct positive relationships
    pos_true_indices = np.where(relations_gt_true != NOF_PREDICATES - 1)
    relations_gt_pos_true = relations_gt_true[pos_true_indices]
    relations_pred_pos_true = relations_pred_true_pos[pos_true_indices]
    results["relationships_pos_correct"] = np.sum(relations_gt_pos_true == relations_pred_pos_true)

    return results


def train(name="test",
          nof_iterations=100,
          learning_rate=0.0001,
          learning_rate_steps=1000,
          learning_rate_decay=0.5,
          load_module_name="module.ckpt",
          use_saved_module=False,
          batch_size=200,
          pred_pos_neg_ratio=10,
          lr_object_coeff=4,
          layers=[500, 500, 500],
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
    gpi_type = "Linguistic",
    including_object = True
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
    if gpu != None:
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
    confidence_relation_ph, confidence_entity_ph, bb_ph, word_embed_relations_ph, word_embed_entities_ph = module.get_in_ph()
    # get labels place holders
    labels_relation_ph, labels_entity_ph, labels_coeff_loss_ph = module.get_labels_ph()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = module.get_module_loss()

    ##
    # get module output
    out_relation_probes, out_entity_probes = module.get_output()

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

        # train images
        vg_train_path = filesmanager.get_file_path("data.visual_genome.train")
        # list of train files
        train_files_list = range(71)
        shuffle(train_files_list)

        validation_files_list = range(71, 73)

        # create one hot vector for predicate_negative (i.e. not labeled)
        relation_neg = np.zeros(NOF_PREDICATES)
        relation_neg[NOF_PREDICATES - 1] = 1

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
            # read data
            file_index = -1
            for file_name in train_files_list:

                file_index += 1

                # load data from file
                file_path = os.path.join(vg_train_path, str(file_name) + ".p")
                file_handle = open(file_path, "rb")
                train_images = cPickle.load(file_handle)
                file_handle.close()
                shuffle(train_images)

                for image in train_images:
                    # set diagonal to be negative predicate (no relation for a single object)
                    indices = np.arange(image.predicates_outputs_with_no_activation.shape[0])
                    image.predicates_outputs_with_no_activation[indices, indices, :] = relation_neg
                    image.predicates_labels[indices, indices, :] = relation_neg

                    # spatial features
                    entity_bb = np.zeros((len(image.objects), 4))
                    for obj_id in range(len(image.objects)):
                        entity_bb[obj_id][0] = image.objects[obj_id].x / 1200.0
                        entity_bb[obj_id][1] = image.objects[obj_id].y / 1200.0
                        entity_bb[obj_id][2] = (image.objects[obj_id].x + image.objects[obj_id].width) / 1200.0
                        entity_bb[obj_id][3] = (image.objects[obj_id].y + image.objects[obj_id].height) / 1200.0

                    # filter non mixed cases
                    relations_neg_labels = image.predicates_labels[:, :, NOF_PREDICATES - 1:]
                    if np.sum(image.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                            relations_neg_labels) == 0:
                        continue

                    if including_object:
                        in_entity_confidence = image.objects_outputs_with_no_activations
                    else:
                        in_entity_confidence = image.objects_labels * 1000

                    # give lower weight to negatives
                    coeff_factor = np.ones(relations_neg_labels.shape)
                    factor = float(np.sum(image.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                        relations_neg_labels) / pred_pos_neg_ratio
                    coeff_factor[relations_neg_labels == 1] *= factor

                    coeff_factor[indices, indices] = 0

                    # create the feed dictionary
                    feed_dict = {confidence_relation_ph: image.predicates_outputs_with_no_activation,
                                 confidence_entity_ph: in_entity_confidence,
                                 bb_ph : entity_bb,
                                 module.phase_ph: True,
                                 word_embed_entities_ph: embed_obj, word_embed_relations_ph: embed_pred,
                                 labels_relation_ph: image.predicates_labels, labels_entity_ph: image.objects_labels,
                                 labels_coeff_loss_ph: coeff_factor.reshape((-1)), module.lr_ph: lr}

                    # run the network
                    out_relation_probes_val, out_entity_probes_val, loss_val, gradients_val = \
                        sess.run([out_relation_probes, out_entity_probes, loss, gradients],
                                 feed_dict=feed_dict)
                    if math.isnan(loss_val):
                        print("NAN")
                        continue

                    # set diagonal to be neg (in order not to take into account in statistics)
                    out_relation_probes_val[indices, indices, :] = relation_neg

                    # append gradient to list (will be applied as a batch of entities)
                    steps.append(gradients_val)

                    # statistic
                    total_loss += loss_val

                    results = test(image.predicates_labels, image.objects_labels, out_relation_probes_val,
                                   out_entity_probes_val)

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
                    obj_accuracy = float(accum_results['entity_correct']) / accum_results['entity_total']
                    predicate_pos_accuracy = float(accum_results['relations_pos_correct']) / accum_results[
                        'relations_pos_total']
                    relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                        'relations_pos_total']
                    logger.log("iter %d.%d - obj %f - pred %f - relation %f" %
                           (epoch, file_index, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy))

            # print stat per epoch
            obj_accuracy = float(accum_results['entity_correct']) / accum_results['entity_total']
            predicate_pos_accuracy = float(accum_results['relations_pos_correct']) / accum_results[
                'relations_pos_total']
            predicate_all_accuracy = float(accum_results['relations_correct']) / accum_results['relations_total']
            relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                'relations_pos_total']
            relationships_all_accuracy = float(accum_results['relationships_correct']) / accum_results[
                'relations_total']

            logger.log("iter %d - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - lr %f" %
                       (epoch, total_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                        predicate_all_accuracy, relationships_all_accuracy, lr))

            # run validation
            if epoch % TEST_ITERATIONS == 0:
                total_test_loss = 0
                accum_test_results = None

                for file_name in validation_files_list:
                    # load data from file
                    file_path = os.path.join(vg_train_path, str(file_name) + ".p")
                    file_handle = open(file_path, "rb")
                    validation_images = cPickle.load(file_handle)
                    file_handle.close()

                    for image in validation_images:
                        # set diagonal to be neg
                        indices = np.arange(image.predicates_outputs_with_no_activation.shape[0])
                        image.predicates_outputs_with_no_activation[indices, indices, :] = relation_neg
                        image.predicates_labels[indices, indices, :] = relation_neg

                        # get shape of extended object to be used by the module
                        extended_confidence_object_shape = np.asarray(image.predicates_outputs_with_no_activation.shape)
                        extended_confidence_object_shape[2] = NOF_OBJECTS

                        # spatial features
                        entity_bb = np.zeros((len(image.objects), 4))
                        for obj_id in range(len(image.objects)):
                            entity_bb[obj_id][0] = image.objects[obj_id].x / 1200.0
                            entity_bb[obj_id][1] = image.objects[obj_id].y / 1200.0
                            entity_bb[obj_id][2] = (image.objects[obj_id].x + image.objects[obj_id].width) / 1200.0
                            entity_bb[obj_id][3] = (image.objects[obj_id].y + image.objects[obj_id].height) / 1200.0

                        # filter non mixed cases
                        relations_neg_labels = image.predicates_labels[:, :, NOF_PREDICATES - 1:]
                        if np.sum(image.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                                relations_neg_labels) == 0:
                            continue

                        # give lower weight to negatives
                        coeff_factor = np.ones(relations_neg_labels.shape)
                        factor = float(np.sum(image.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                            relations_neg_labels) / pred_pos_neg_ratio
                        coeff_factor[relations_neg_labels == 1] *= factor
                        coeff_factor[indices, indices] = 0
                        coeff_factor[relations_neg_labels == 1] = 0

                        if including_object:
                            in_entity_confidence = image.objects_outputs_with_no_activations
                        else:
                            in_entity_confidence = image.objects_labels * 1000

                        # create the feed dictionary
                        feed_dict = {confidence_relation_ph: image.predicates_outputs_with_no_activation,
                                     confidence_entity_ph: in_entity_confidence,
                                     module.entity_bb_ph: entity_bb,
                                     module.word_embed_entities_ph: embed_obj,
                                     module.phase_ph: False,
                                     module.word_embed_relations_ph: embed_pred,
                                     labels_relation_ph: image.predicates_labels,
                                     labels_entity_ph: image.objects_labels,
                                     labels_coeff_loss_ph: coeff_factor.reshape((-1))}

                        # run the network
                        out_relation_probes_val, out_entity_probes_val, loss_val = sess.run(
                            [out_relation_probes, out_entity_probes, loss],
                            feed_dict=feed_dict)

                        # set diagonal to be neg (in order not to take into account in statistics)
                        out_relation_probes_val[indices, indices, :] = relation_neg

                        # statistic
                        total_test_loss += loss_val

                        # statistics
                        results = test(image.predicates_labels, image.objects_labels,
                                       out_relation_probes_val, out_entity_probes_val)

                        # accumulate results
                        if accum_test_results is None:
                            accum_test_results = results
                        else:
                            for key in results:
                                accum_test_results[key] += results[key]


                # print stat
                obj_accuracy = float(accum_test_results['entity_correct']) / accum_test_results['entity_total']
                predicate_pos_accuracy = float(accum_test_results['relations_pos_correct']) / accum_test_results[
                    'relations_pos_total']
                predicate_all_accuracy = float(accum_test_results['relations_correct']) / accum_test_results[
                    'relations_total']
                relationships_pos_accuracy = float(accum_test_results['relationships_pos_correct']) / \
                                             accum_test_results[
                                                 'relations_pos_total']
                relationships_all_accuracy = float(accum_test_results['relationships_correct']) / accum_test_results[
                    'relations_total']

                logger.log("TEST - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f" %
                           (total_test_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                            predicate_all_accuracy, relationships_all_accuracy))

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
