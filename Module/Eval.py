import inspect
import os
import sys

sys.path.append("..")

from FilesManager.FilesManager import FilesManager
from Module import Module
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
from Train import test, NOF_OBJECTS, NOF_PREDICATES
import cPickle
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
import time


def eval_image(labels_relation, labels_entity, out_confidence_relation_val, out_confidence_entity_val, k=100):
    """
    Scene Graph Classification -
    R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet prediction in an image)
    :param labels_relation: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_entity: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_confidence_relation_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_confidence_entity_val: confidence of image objects - shape (N, NOF_OBJECTS)
    :param k: k most confident predictions to consider
    :return: image score, number of the gt triplets that appear in the k most confident predictions,
                         number of the gt triplets
    """
    # iterate over each relation to predict and find k highest predictions
    top_predictions = np.zeros((0, 6))

    # results per relation
    per_relation_correct = np.zeros(NOF_PREDICATES)
    per_relation_total = np.zeros(NOF_PREDICATES)

    N = labels_entity.shape[0]
    if N == 1:
        return 0, 0, 0, per_relation_correct, per_relation_total

    relation_pred = np.argmax(out_confidence_relation_val[:, :, :NOF_PREDICATES - 1], axis=2)
    relation_scores = np.max(out_confidence_relation_val[:, :, :NOF_PREDICATES - 1], axis=2)
    entity_pred = np.argmax(out_confidence_entity_val, axis=1)
    entity_scores = np.max(out_confidence_entity_val, axis=1)

    # get list of the top k most confident triplets predictions
    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if subject_index == object_index:
                continue

            # create entry with the scores
            triplet_prediction = np.zeros((1, 6))
            triplet_prediction[0][0] = subject_index
            triplet_prediction[0][1] = object_index
            triplet_prediction[0][2] = entity_pred[subject_index]
            triplet_prediction[0][3] = relation_pred[subject_index][object_index]
            triplet_prediction[0][4] = entity_pred[object_index]
            triplet_prediction[0][5] = relation_scores[subject_index][object_index] * entity_scores[subject_index] * \
                                       entity_scores[object_index]

            # append to the list of highest predictions
            top_predictions = np.concatenate((top_predictions, triplet_prediction))

    # get k highest confidence
    top_k_indices = np.argsort(top_predictions[:, 5])[-k:]
    global_sub_ids = top_predictions[top_k_indices, 0]
    global_obj_ids = top_predictions[top_k_indices, 1]
    sub_pred = top_predictions[top_k_indices, 2]
    relation_pred = top_predictions[top_k_indices, 3]
    obj_pred = top_predictions[top_k_indices, 4]

    relations_gt = np.argmax(labels_relation, axis=2)
    entities_gt = np.argmax(labels_entity, axis=1)

    img_score = 0
    nof_pos_relationship = 0
    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if subject_index == object_index:
                continue
            # filter negative relationship
            if relations_gt[subject_index, object_index] == NOF_PREDICATES - 1:
                continue

            predicate_id = relations_gt[subject_index][object_index]
            sub_id = entities_gt[subject_index]
            obj_id = entities_gt[object_index]

            nof_pos_relationship += 1
            per_relation_total[predicate_id] += 1

            # filter the predictions for the specific subject
            sub_indices = set(np.where(global_sub_ids == subject_index)[0])
            obj_indices = set(np.where(global_obj_ids == object_index)[0])
            sub_pred_indices = set(np.where(sub_pred == sub_id)[0])
            predicate_pred_indices = set(np.where(relation_pred == predicate_id)[0])
            obj_pred_indices = set(np.where(obj_pred == obj_id)[0])

            indices = sub_indices & obj_indices & sub_pred_indices & obj_pred_indices & predicate_pred_indices
            if len(indices) != 0:
                img_score += 1
                per_relation_correct[predicate_id] += 1
            else:
                img_score = img_score

    if nof_pos_relationship != 0:
        img_score_percent = float(img_score) / nof_pos_relationship
    else:
        img_score_percent = 0

    return img_score_percent, img_score, nof_pos_relationship, per_relation_correct, per_relation_total


def eval(load_module_name=None, k=100, layers=[500, 500, 500], gpu=1):
    """
    Evaluate module:
    - Scene Graph Classification - R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet prediction in an image)
    :param load_module_name: name of the module to load
    :param k: see description
    :param layers: hidden layers of relation and entity classifier
    :param gpu: gpu number to use
    :return: nothing - output to logger instead
    """
    gpi_type = "Linguistic"
    k_recall = True
    filesmanager = FilesManager()
    # create logger
    logger = Logger()

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
                    is_train=False, layers=layers, including_object=True)

    # get input place holders
    confidence_relation_ph, confidence_entity_ph, bb_ph, word_embed_relations_ph, word_embed_entities_ph = module.get_in_ph()
    # get module output
    out_relation_probes, out_entity_probes = module.get_output()

    # Initialize the Computational Graph
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # read data
    entities_path = filesmanager.get_file_path("data.visual_genome.test")

    test_files_list = range(35)

    # embeddings
    embed_obj = FilesManager().load_file("language_module.word2vec.object_embeddings")
    embed_pred = FilesManager().load_file("language_module.word2vec.predicate_embeddings")
    embed_pred = np.concatenate((embed_pred, np.zeros(embed_pred[:1].shape)), axis=0)  # concat negative represntation
    accum_results = None
    with tf.Session() as sess:
        if load_module_name is not None:
            # Restore variables from disk.
            module_path = filesmanager.get_file_path("sg_module.train.saver")
            module_path_load = os.path.join(module_path, load_module_name + "_module.ckpt")
            if os.path.exists(module_path_load + ".index"):
                saver.restore(sess, module_path_load)
                logger.log("Model restored.")
            else:
                raise Exception("Module not found")
        else:
            sess.run(init)
        # eval module

        nof = 0
        total = 0
        correct_all = 0
        total_all = 0

        # create one hot vector for null relation
        relation_neg = np.zeros(NOF_PREDICATES)
        relation_neg[NOF_PREDICATES - 1] = 1
        
        index = 0

        for file_name in test_files_list:
            file_path = os.path.join(entities_path, str(file_name) + ".p")
            file_handle = open(file_path, "rb`")
            test_entities = cPickle.load(file_handle)
            file_handle.close()
            for entity in test_entities:

                # set diagonal to be negative relation
                N = entity.predicates_outputs_with_no_activation.shape[0]
                indices = np.arange(N)
                entity.predicates_outputs_with_no_activation[indices, indices, :] = relation_neg
                entity.predicates_labels[indices, indices, :] = relation_neg

                # create bounding box info per object
                obj_bb = np.zeros((len(entity.objects), 4))
                for obj_id in range(len(entity.objects)):
                    obj_bb[obj_id][0] = entity.objects[obj_id].x / 1200.0
                    obj_bb[obj_id][1] = entity.objects[obj_id].y / 1200.0
                    obj_bb[obj_id][2] = (entity.objects[obj_id].x + entity.objects[obj_id].width) / 1200.0
                    obj_bb[obj_id][3] = (entity.objects[obj_id].y + entity.objects[obj_id].height) / 1200.0

                # filter images with no positive relations
                relations_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
                if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                        relations_neg_labels) == 0:
                    continue

                # use object class labels for pred class (multiply be some factor to convert to confidence)
                in_entity_confidence = entity.objects_outputs_with_no_activations

                # create the feed dictionary
                feed_dict = {confidence_relation_ph: entity.predicates_outputs_with_no_activation,
                             confidence_entity_ph: in_entity_confidence,
                             bb_ph: obj_bb,
                             module.phase_ph: False,
                             word_embed_entities_ph: embed_obj, word_embed_relations_ph: embed_pred}

                out_relation_probes_val, out_entity_probes_val = \
                    sess.run([out_relation_probes, out_entity_probes],
                             feed_dict=feed_dict)

                out_relation_probes_val[indices, indices, :] = relation_neg

                results = test(entity.predicates_labels, entity.objects_labels, out_relation_probes_val,
                               out_entity_probes_val)

                # accumulate results
                if accum_results is None:
                    accum_results = results
                else:
                    for key in results:
                        accum_results[key] += results[key]

                # eval image
                k_metric_res, correct_image, total_image, img_per_relation_correct, img_per_relation_total = eval_image(
                    entity.predicates_labels,
                    entity.objects_labels, out_relation_probes_val, out_entity_probes_val, k=min(k, N * N - N))
                # filter images without positive relations
                if total_image == 0:
                    continue

                nof += 1
                total += k_metric_res
                total_score = float(total) / nof
                correct_all += correct_image
                total_all += total_image
                logger.log("\rresult %d - %f (%d / %d) - total %f (%d)" % (
                    index, k_metric_res, correct_image, total_image, total_score, entity.image.id))

                index += 1

            relation_accuracy = float(accum_results['entity_correct']) / accum_results['entity_total']
            relation_pos_accuracy = float(accum_results['relations_pos_correct']) / accum_results[
                'relations_pos_total']
            relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                'relations_pos_total']
            logger.log("entity %f - positive relation %f - positive triplet %f" %
                       (relation_accuracy, relation_pos_accuracy, relationships_pos_accuracy))

            time.sleep(3)


        logger.log("(%s) Final Result for k=%d - %f" % (load_module_name, k, total_score))


if __name__ == "__main__":
    k_recall = True
    gpu = 2
    layers = [500, 500, 500]
    
    #load_module_name = "gpi_ling_atten_rnn2_sdo2_best"
    load_module_name = "gpi_ling_atten_rnn2_sdo_new_files_withoutf_sgd_best"
    load_module_name = "gpi_ling_orig_best"
    k = 100
    eval(load_module_name, k, layers, gpu)
    exit()
