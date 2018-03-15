import inspect
import os
import sys

sys.path.append("..")

import itertools
from FilesManager.FilesManager import FilesManager
from Module import Module
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
from Train import test, predicate_class_recall, NOF_OBJECTS, NOF_PREDICATES
import cPickle
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
#import matplotlib.pyplot as plt
import time


def eval_image(labels_predicate, labels_object, out_confidence_predicate_val, out_confidence_object_val, k=100):
    """
    Scene Graph Classification -
    R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet prediction in an image)
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_object: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_confidence_predicate_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_confidence_object_val: confidence of image objects - shape (N, NOF_OBJECTS)
    :param k: k most confident predictions to consider
    :return: image score, number of the gt triplets that appear in the k most confident predictions,
                         number of the gt triplets
    """
    # iterate over each relation to predict and find k highest predictions
    top_predictions = np.zeros((0, 6))

    # results per predicate
    per_predicate_correct = np.zeros(NOF_PREDICATES)
    per_predicate_total = np.zeros(NOF_PREDICATES)

    N = labels_object.shape[0]
    if N == 1:
        return 0, 0, 0, per_predicate_correct, per_predicate_total

    predicate_pred = np.argmax(out_confidence_predicate_val[:, :, :NOF_PREDICATES - 1], axis=2)
    predicate_scores = np.max(out_confidence_predicate_val[:, :, :NOF_PREDICATES - 1], axis=2)
    object_pred = np.argmax(out_confidence_object_val, axis=1)
    object_scores = np.max(out_confidence_object_val, axis=1)

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
            triplet_prediction[0][2] = object_pred[subject_index]
            triplet_prediction[0][3] = predicate_pred[subject_index][object_index]
            triplet_prediction[0][4] = object_pred[object_index]
            triplet_prediction[0][5] = predicate_scores[subject_index][object_index] * object_scores[subject_index] * \
                                       object_scores[object_index]

            # append to the list of highest predictions
            top_predictions = np.concatenate((top_predictions, triplet_prediction))

    # get k highest confidence
    top_k_indices = np.argsort(top_predictions[:, 5])[-k:]
    global_sub_ids = top_predictions[top_k_indices, 0]
    global_obj_ids = top_predictions[top_k_indices, 1]
    sub_pred = top_predictions[top_k_indices, 2]
    predicate_pred = top_predictions[top_k_indices, 3]
    obj_pred = top_predictions[top_k_indices, 4]

    predicates_gt = np.argmax(labels_predicate, axis=2)
    objects_gt = np.argmax(labels_object, axis=1)

    img_score = 0
    nof_pos_relationship = 0
    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if subject_index == object_index:
                continue
            # filter negative relationship
            if predicates_gt[subject_index, object_index] == NOF_PREDICATES - 1:
                continue

            predicate_id = predicates_gt[subject_index][object_index]
            sub_id = objects_gt[subject_index]
            obj_id = objects_gt[object_index]

            nof_pos_relationship += 1
            per_predicate_total[predicate_id] += 1

            # filter the predictions for the specific subject
            sub_indices = set(np.where(global_sub_ids == subject_index)[0])
            obj_indices = set(np.where(global_obj_ids == object_index)[0])
            sub_pred_indices = set(np.where(sub_pred == sub_id)[0])
            predicate_pred_indices = set(np.where(predicate_pred == predicate_id)[0])
            obj_pred_indices = set(np.where(obj_pred == obj_id)[0])

            indices = sub_indices & obj_indices & sub_pred_indices & obj_pred_indices & predicate_pred_indices
            if len(indices) != 0:
                img_score += 1
                per_predicate_correct[predicate_id] += 1
            else:
                img_score = img_score

    if nof_pos_relationship != 0:
        img_score_percent = float(img_score) / nof_pos_relationship
    else:
        img_score_percent = 0

    return img_score_percent, img_score, nof_pos_relationship, per_predicate_correct, per_predicate_total


def eval_image_easy(labels_predicate, labels_object,
                    out_confidence_predicate_val, out_confidence_object_val, k=100):
    """
    Scene Graph Classification -
    R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet prediction in an image)
    :param img:
    :param reverse_object_ids:
    :param reverse_predicate_ids:
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_object: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_confidence_predicate_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_confidence_object_val: confidence of image objects - shape (N, NOF_OBJECTS)
    :param k: k most confident predictions to consider
    :return: image score, number of the gt triplets that appear in the k most confident predictions,
                         number of the gt triplets
    """

    # iterate over each relation to predict and find k highest predictions
    top_predictions = np.zeros((0,))
    top_likelihoods = np.zeros((0,))
    top_k_global_subject_ids = np.zeros((0,))
    top_k_global_object_ids = np.zeros((0,))

    N = labels_object.shape[0]
    if N == 1:
        return 0, 0, 0

    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if subject_index == object_index:
                continue

            predicate_prob = out_confidence_predicate_val[subject_index][object_index]
            subject_prob = out_confidence_object_val[subject_index]
            object_prob = out_confidence_object_val[object_index]

            # calc tensor of probabilities of visual module
            predict_prob = np.multiply.outer(subject_prob, np.multiply.outer(predicate_prob.flatten(), object_prob))

            # remove negative probabilities
            predict_prob[:, NOF_PREDICATES - 1, :] = 0

            # get the highset probabilities
            max_k_predictions = np.argpartition(predict_prob.flatten(), -k)[-k:]
            max_k_predictions_triplets = np.unravel_index(max_k_predictions, predict_prob.shape)
            max_k_subjects = max_k_predictions_triplets[0]
            max_k_predicates = max_k_predictions_triplets[1]
            max_k_objects = max_k_predictions_triplets[2]
            max_k_likelihoods = predict_prob[max_k_subjects, max_k_predicates, max_k_objects]

            # append to the list of highest predictions
            top_predictions = np.concatenate((top_predictions, max_k_predictions))
            top_likelihoods = np.concatenate((top_likelihoods, max_k_likelihoods))

            # store the relevant subject and object
            max_k_global_subject_ids = np.ones(max_k_likelihoods.shape) * subject_index
            max_k_global_object_ids = np.ones(max_k_likelihoods.shape) * object_index
            top_k_global_subject_ids = np.concatenate((top_k_global_subject_ids, max_k_global_subject_ids))
            top_k_global_object_ids = np.concatenate((top_k_global_object_ids, max_k_global_object_ids))

    # get k highest confidence
    top_k_indices = np.argsort(top_likelihoods)[-k:]
    predictions = top_predictions[top_k_indices]
    global_sub_ids = top_k_global_subject_ids[top_k_indices]
    global_obj_ids = top_k_global_object_ids[top_k_indices]
    likelihoods = top_likelihoods[top_k_indices]
    triplets = np.unravel_index(predictions.astype(int), predict_prob.shape)

    predicates_gt = np.argmax(labels_predicate, axis=2)
    objects_gt = np.argmax(labels_object, axis=1)

    img_score = 0
    nof_pos_relationship = 0
    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if subject_index == object_index:
                continue
            # filter negative relationship
            if predicates_gt[subject_index, object_index] == NOF_PREDICATES - 1:
                continue

            nof_pos_relationship += 1
            predicate_id = predicates_gt[subject_index][object_index]
            sub_id = objects_gt[subject_index]
            obj_id = objects_gt[object_index]
            gt_relation = np.ravel_multi_index((sub_id, predicate_id, obj_id), predict_prob.shape)

            # filter the predictions for the specific subject
            sub_predictions_indices = set(np.where(global_sub_ids == subject_index)[0])
            obj_predictions_indices = set(np.where(global_obj_ids == object_index)[0])
            relation_indices = set(np.where(predictions == gt_relation)[0])

            indices = sub_predictions_indices & obj_predictions_indices & relation_indices
            if len(indices) != 0:
                img_score += 1
            else:
                img_score = img_score

    if nof_pos_relationship != 0:
        img_score_precent = float(img_score) / nof_pos_relationship
    else:
        img_score_precent = 0

    return img_score_precent, img_score, nof_pos_relationship, np.ones(NOF_PREDICATES), np.ones(NOF_PREDICATES)


def eval(load_module_name=None, gpi_type="Linguistic" ,k_recall=True, pred_class=True, rnn_steps=0, k=100,
         layers=[500, 500, 500], gpu=1):
    """
    Evaluate module:
    - Scene Graph Classification - R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet prediction in an image)
    - Predicate Classification - Examine the model performance on predicates classification in isolation from other factors
    :param load_module_name: name of the module to load
    :param gpi_type:
    :param k_recall:
    :param pred_class:
    :param rnn_steps:
    :param k:
    :param layers:
    :param gpu: gpu number to use
    :return: nothing - output to logger instead
    """
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.log("os.environ[\"CUDA_VISIBLE_DEVICES\"] = " + str(gpu))

    # create module
    module = Module(gpi_type=gpi_type, nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                    is_train=False,
                    rnn_steps=rnn_steps, layers=layers, including_object=True)

    # get input place holders
    confidence_relation_ph, confidence_entity_ph = module.get_in_ph()
    # get module output
    out_predicate_probes, out_object_probes = module.get_output()

    # Initialize the Computational Graph
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # read data
    entities_path = filesmanager.get_file_path("data.visual_genome.detections_v4")

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

    # TBD: get ids
    object_ids = []
    predicate_ids = []
    reverse_object_ids = {object_ids[id]: id for id in object_ids}
    reverse_predicate_ids = {predicate_ids[id]: id for id in predicate_ids}

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
        correct_predicate = np.zeros(NOF_PREDICATES)
        total_predicate = np.zeros(NOF_PREDICATES)
        nof = 0
        total = 0
        correct_all = 0
        total_all = 0

        # create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1
        index = 0
        # results per predicate
        per_predicate_correct = np.zeros(NOF_PREDICATES)
        per_predicate_total = np.zeros(NOF_PREDICATES)
        for file_name in test_files_list:
            file_path = os.path.join(entities_path, file_name)
            file_handle = open(file_path, "rb`")
            test_entities = cPickle.load(file_handle)
            file_handle.close()
            for entity in test_entities:

                # set diagonal to be neg
                N = entity.predicates_outputs_with_no_activation.shape[0]
                indices = np.arange(N)
                entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
                entity.predicates_labels[indices, indices, :] = predicate_neg
                entity.predicates_probes[indices, indices, :] = predicate_neg

                # create bounding box info per object
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

                # use object class labels for pred class (multiply be some factor to convert to confidence)
                if pred_class:
                    in_object_confidence = entity.objects_labels * 1000
                else:
                    in_object_confidence = entity.objects_outputs_with_no_activations

                # create the feed dictionary
                feed_dict = {confidence_relation_ph: entity.predicates_outputs_with_no_activation,
                             confidence_entity_ph: in_object_confidence,
                             module.entity_bb_ph: obj_bb,
                             module.word_embed_entities_ph: embed_obj, module.word_embed_relations_ph: embed_pred}

                out_predicate_probes_val, out_object_probes_val = \
                    sess.run([out_predicate_probes, out_object_probes],
                             feed_dict=feed_dict)

                # Sanity check (uncomment to get Label Predictor)
                # out_predicate_probes_val = entity.predicates_probes
                # out_object_probes_val = entity.objects_probs
                out_predicate_probes_val[indices, indices, :] = predicate_neg
                # use object class labels for pred class                
                if pred_class:
                    out_object_probes_val = entity.objects_labels

                results = test(entity.predicates_labels, entity.objects_labels, out_predicate_probes_val,
                               out_object_probes_val)
                results_orig = test(entity.predicates_labels, entity.objects_labels, entity.predicates_probes,
                                    entity.objects_probs)

                predicates_pos_accuracy = float(results['predicates_pos_correct']) / results['predicates_pos_total']
                predicates_pos_accuracy_orig = float(results_orig['predicates_pos_correct']) / results_orig[
                    'predicates_pos_total']
                obj_accuracy = float(results['obj_correct']) / results['obj_total']
                obj_accuracy_orig = float(results_orig['obj_correct']) / results_orig['obj_total']

                # accumulate results
                if accum_results is None:
                    accum_results = results
                else:
                    for key in results:
                        accum_results[key] += results[key]

                # eval image
                if k_recall:
                    k_metric_res, correct_image, total_image, img_per_pred_correct, img_per_pred_total = eval_image(
                        entity.predicates_labels,
                        entity.objects_labels, out_predicate_probes_val, out_object_probes_val, k=min(k, N * N - N))
                    # filter images without positive relations
                    if total_image == 0:
                        continue

                    nof += 1
                    total += k_metric_res
                    total_score = float(total) / nof
                    correct_all += correct_image
                    total_all += total_image
                    score_all = float(correct_all) / total_all
                    per_predicate_correct += img_per_pred_correct
                    per_predicate_total += img_per_pred_total
                    logger.log("\rresult %d - %f (%d / %d) - total %f - %f (%d)" % (
                        index, k_metric_res, correct_image, total_image, total_score, score_all, entity.image.id))

                # eval per predicate
                correct_predicate_image, total_predicate_image = predicate_class_recall(entity.predicates_labels,
                                                                                        out_predicate_probes_val)
                correct_predicate += correct_predicate_image
                total_predicate += total_predicate_image

                index += 1

            obj_accuracy = float(accum_results['obj_correct']) / accum_results['obj_total']
            predicate_pos_accuracy = float(accum_results['predicates_pos_correct']) / accum_results[
                'predicates_pos_total']
            relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                'predicates_pos_total']
            logger.log("obj %f - pred %f - relation %f" %
                       (obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy))
            time.sleep(3)

        total_predicate = per_predicate_total
        correct_predicate = per_predicate_correct
        for i in range(NOF_PREDICATES):
            if total_predicate[i] != 0:
                logger.log("{0} recall@5 is {1} (total - {2}, correct {3})".format(reverse_predicate_ids[i],
                                                                                   float(correct_predicate[i]) /
                                                                                   total_predicate[i],
                                                                                   total_predicate[i],
                                                                                   correct_predicate[i]))

        logger.log("Final Result for pred_class=%s k=%d - %f" % (str(pred_class), k, total_score))


if __name__ == "__main__":
    k_recall = True
    rnn_steps = 2
    gpu = 3
    layers = [500, 500, 500]
    #layers = [200, 200]
    gpi_type="FeatureAttention"
    gpi_type="Linguistic"
    #gpi_type="NeighbourAttention"
    load_module_name = "gpi_ling_atten_rnn2_exact_best"
    #load_module_name = "gpi_feature_atten_2_200_best"
    #load_module_name = "_best"
    # rnn2_bb_all_short_best, True , [1000, 200]
    pred_class = False
    k = 100
    eval(load_module_name, gpi_type, k_recall, pred_class, rnn_steps, k, layers, gpu)
    exit()
