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

def iou(box_a, box_b):
    union_area = (max(box_a[2], box_b[2]) - min(box_a[0], box_b[0]) + 1) * (max(box_a[3], box_b[3]) - min(box_a[1], box_b[1]) + 1) 
    overlap_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + 1
    if overlap_w <= 0:
        return 0
    overlap_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + 1
    if overlap_h <= 0:
        return 0
    return float(overlap_w * overlap_h) / union_area

def eval_image(entity, labels_relation, labels_entity, out_confidence_relation_val, out_confidence_entity_val, k=100):
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
    top_predictions = np.zeros((0, 12))

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
            triplet_prediction = np.zeros((1, 12))
            triplet_prediction[0][0] = entity.objects[subject_index].x
            triplet_prediction[0][1] = entity.objects[subject_index].y
            triplet_prediction[0][2] = entity.objects[subject_index].x + entity.objects[subject_index].width
            triplet_prediction[0][3] = entity.objects[subject_index].y + entity.objects[subject_index].height
            triplet_prediction[0][4] = entity.objects[object_index].x
            triplet_prediction[0][5] = entity.objects[object_index].y
            triplet_prediction[0][6] = entity.objects[object_index].x + entity.objects[object_index].width
            triplet_prediction[0][7] = entity.objects[object_index].y + entity.objects[object_index].height
            
            triplet_prediction[0][8] = entity_pred[subject_index]
            triplet_prediction[0][9] = relation_pred[subject_index][object_index]
            triplet_prediction[0][10] = entity_pred[object_index]
            triplet_prediction[0][11] = relation_scores[subject_index][object_index] * entity_scores[subject_index] * \
                                       entity_scores[object_index]

            # append to the list of highest predictions
            top_predictions = np.concatenate((top_predictions, triplet_prediction))

    # get k highest confidence
    top_k_indices = np.argsort(top_predictions[:, 11])[-k:]
    sub_boxes = top_predictions[top_k_indices, :4]
    obj_boxes = top_predictions[top_k_indices, 4:8]
    sub_pred = top_predictions[top_k_indices, 8]
    relation_pred = top_predictions[top_k_indices, 9]
    obj_pred = top_predictions[top_k_indices, 10]

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

            gt_sub_box = np.zeros((4))
            gt_sub_box[0] = entity.objects[subject_index].x
            gt_sub_box[1] = entity.objects[subject_index].y
            gt_sub_box[2] = entity.objects[subject_index].x + entity.objects[subject_index].width
            gt_sub_box[3] = entity.objects[subject_index].y + entity.objects[subject_index].height

            gt_obj_box = np.zeros((4))
            gt_obj_box[0] = entity.objects[object_index].x
            gt_obj_box[1] = entity.objects[object_index].y
            gt_obj_box[2] = entity.objects[object_index].x + entity.objects[object_index].width
            gt_obj_box[3] = entity.objects[object_index].y + entity.objects[object_index].height

            predicate_id = relations_gt[subject_index][object_index]
            sub_id = entities_gt[subject_index]
            obj_id = entities_gt[object_index]

            nof_pos_relationship += 1
            per_relation_total[predicate_id] += 1

            # filter according to iou 
            found = False
            for top_k_i in range(k):
                if sub_id != sub_pred[top_k_i] or obj_id != obj_pred[top_k_i] or predicate_id !=relation_pred[top_k_i]:
                    continue
                iou_sub_val = iou(gt_sub_box, sub_boxes[top_k_i])
                if iou_sub_val < 0.5:
                    continue
                iou_obj_val = iou(gt_obj_box, obj_boxes[top_k_i])
                if iou_obj_val < 0.5:
                    continue
                
                found = True
                break
 
            if found:
                img_score += 1
                per_relation_correct[predicate_id] += 1
            else:
                img_score = img_score

    if nof_pos_relationship != 0:
        img_score_percent = float(img_score) / float(nof_pos_relationship)
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

    # print eval params
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
            if load_module_name=="gpi_linguistic_pretrained":
                module_path = os.path.join(filesmanager.get_file_path("data.visual_genome.data"), "data")
            else:
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
        basline_path = filesmanager.get_file_path("data.visual_genome.test_baseline")
        for file_name in test_files_list:
            file_path = os.path.join(entities_path, str(file_name) + ".p")
            file_handle = open(file_path, "rb")
            test_entities = cPickle.load(file_handle)
            file_handle.close()

            for entity in test_entities:
                file_path = os.path.join(basline_path, str(entity.image.id) + ".p")
                if not os.path.exists(file_path):
                    continue
                file_handle = open(file_path, "rb")
                detector_data = cPickle.load(file_handle)
                file_handle.close()

                entity.predicates_outputs_with_no_activation = detector_data["rel_dist_mapped"]
                entity.objects_outputs_with_no_activations = detector_data["obj_dist_mapped"]
                # set diagonal to be negative relation
                N = entity.predicates_outputs_with_no_activation.shape[0]
                indices = np.arange(N)
                entity.predicates_outputs_with_no_activation[indices, indices, :] = relation_neg
                entity.predicates_labels[indices, indices, :] = relation_neg

                # create bounding box info per object
                obj_bb = np.zeros((len(entity.objects), 14))
                for obj_id in range(len(entity.objects)):
                    obj_bb[obj_id][0] = entity.objects[obj_id].x / 1200.0
                    obj_bb[obj_id][1] = entity.objects[obj_id].y / 1200.0
                    obj_bb[obj_id][2] = (entity.objects[obj_id].x + entity.objects[obj_id].width) / 1200.0
                    obj_bb[obj_id][3] = (entity.objects[obj_id].y + entity.objects[obj_id].height) / 1200.0
                    obj_bb[obj_id][4] = entity.objects[obj_id].x
                    obj_bb[obj_id][5] = -1 * entity.objects[obj_id].x
                    obj_bb[obj_id][6] = entity.objects[obj_id].y
                    obj_bb[obj_id][7] = -1 * entity.objects[obj_id].y 
                    obj_bb[obj_id][8] = entity.objects[obj_id].width * entity.objects[obj_id].height
                    obj_bb[obj_id][9] = -1 * entity.objects[obj_id].width * entity.objects[obj_id].height                     
                obj_bb[:, 4] = np.argsort(obj_bb[:, 4])
                obj_bb[:, 5] = np.argsort(obj_bb[:, 5])
                obj_bb[:, 6] = np.argsort(obj_bb[:, 6])
                obj_bb[:, 7] = np.argsort(obj_bb[:, 7])
                obj_bb[:, 8] = np.argsort(obj_bb[:, 8])
                obj_bb[:, 9] = np.argsort(obj_bb[:, 9])
                obj_bb[:, 10] = np.argsort(np.max(entity.objects_outputs_with_no_activations, axis=1))
                obj_bb[:, 11] = np.argsort(-1 * np.max(entity.objects_outputs_with_no_activations, axis=1))
                obj_bb[:, 12] = np.arange(obj_bb.shape[0])
                obj_bb[:, 13] = np.arange(obj_bb.shape[0], 0, -1)

                # filter images with no positive relations
                relations_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
                if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 1]) == 0:
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
                k_metric_res, correct_image, total_image, img_per_relation_correct, img_per_relation_total = eval_image(entity,
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
    gpu = 1
    layers = [500, 500, 500]
    
    load_module_name = "gpi_linguistic_pretrained"
    k = 100
    eval(load_module_name, k, layers, gpu)
    exit()
