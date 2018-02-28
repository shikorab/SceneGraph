import tensorflow as tf
import sys

sys.path.append("..")
from Utils.Logger import Logger


class Module(object):
    """
    RNN Module which gets as an input the confidence of predicates and objects
    and outputs an improved confidence for predicates and objects
    """

    def __init__(self, gpi_type="Linguistic", nof_predicates=51, nof_objects=150, rnn_steps=2, is_train=True,
                 learning_rate=0.0001,
                 learning_rate_steps=120, learning_rate_decay=0.5,
                 including_object=False, layers=[500, 500, 500], reg_factor=0.0, lr_object_coeff=4):
        """
        Construct module:
        - create input placeholders
        - create rnn step
        - apply SGP rnn_steps times
        - create labels placeholders
        - create module loss and train_step

        :param nof_predicates: nof predicate labels
        :param nof_objects: nof object labels
        :param rnn_steps: rnn length
        :param is_train: whether the module will be used to train or eval
        """
        # save input
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate = learning_rate
        self.nof_predicates = nof_predicates
        self.nof_objects = nof_objects
        self.is_train = is_train
        self.rnn_steps = rnn_steps
        self.embed_size = 300
        self.gpi_type = gpi_type

        self.including_object = including_object
        self.lr_object_coeff = lr_object_coeff
        self.layers = layers
        self.reg_factor = reg_factor
        self.activation_fn = tf.nn.relu
        self.reuse = False
        # logging module
        logger = Logger()

        ##
        # module input
        self.phase_ph = tf.placeholder(tf.bool, name='phase')

        # confidence
        self.confidence_relation_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                     name="confidence_relation")
        self.confidence_entity_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects),
                                                   name="confidence_entity")

        # spatial features
        N = tf.slice(tf.shape(self.confidence_relation_ph), [0], [1], name="N")
        self.extended_confidence_entity_shape = tf.concat((N, tf.shape(self.confidence_entity_ph)), 0)
        self.entity_bb_ph = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="obj_bb")
        self.extended_obj_bb_shape = tf.concat((N, tf.shape(self.entity_bb_ph)), 0)
        self.expand_obj_bb = tf.add(tf.zeros(self.extended_obj_bb_shape), self.entity_bb_ph, name="expand_obj_bb")
        # expand subject bb
        expand_sub_bb = tf.transpose(self.expand_obj_bb, perm=[1, 0, 2], name="expand_sub_bb")
        self.expand_sub_bb = expand_sub_bb
        self.bb_features = tf.concat((expand_sub_bb, self.expand_obj_bb), axis=2, name="bb_features")

        # word embeddings
        self.word_embed_entities_ph = tf.placeholder(dtype=tf.float32, shape=(self.nof_objects, self.embed_size),
                                                     name="word_embed_objects")
        self.word_embed_relations_ph = tf.placeholder(dtype=tf.float32, shape=(self.nof_predicates, self.embed_size),
                                                      name="word_embed_predicates")

        # labels
        if self.is_train:
            self.labels_relation_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                     name="labels_predicate")
            self.labels_entity_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects),
                                                   name="labels_object")
            self.labels_coeff_loss_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="labels_coeff_loss")

        # store all the outputs of of rnn steps
        self.out_confidence_entity_lst = []
        self.out_confidence_relation_lst = []
        # rnn stage module
        confidence_relation = self.confidence_relation_ph
        confidence_entity = self.confidence_entity_ph

        # features msg
        for step in range(self.rnn_steps):
            confidence_relation, confidence_entity_temp = \
                self.sgp(in_confidence_relation=confidence_relation,
                         in_confidence_entity=confidence_entity,
                         scope_name="deep_graph")
            # store the confidence
            self.out_confidence_relation_lst.append(confidence_relation)
            if self.including_object:
                confidence_entity = confidence_entity_temp
                # store the confidence
                self.out_confidence_entity_lst.append(confidence_entity_temp)

        self.out_confidence_relation = confidence_relation
        self.out_confidence_entity = confidence_entity
        reshaped_relation_confidence = tf.reshape(confidence_relation, (-1, self.nof_predicates))
        self.reshaped_relation_probes = tf.nn.softmax(reshaped_relation_confidence)
        self.out_relation_probes = tf.reshape(self.reshaped_relation_probes, tf.shape(confidence_relation),
                                              name="out_relation_probes")
        self.out_entity_probes = tf.nn.softmax(confidence_entity, name="out_entity_probes")

        # loss
        if self.is_train:
            # Learning rate
            self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")

            self.loss, self.gradients, self.grad_placeholder, self.train_step = self.module_loss()

    def nn(self, features, layers, out, scope_name, seperated_layer=False, last_activation=None):
        """
        simple nn to convert features to confidence
        :param features: features tensor
        :param layers:
        :param out: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :return: confidence
        """
        with tf.variable_scope(scope_name):

            # first layer each feature seperatly
            features_h_lst = []
            index = 0
            scope = None
            for feature in features:
                if seperated_layer:
                    in_size = feature.shape[-1]._value
                    if self.reuse:
                        scope = str(index)
                    h = tf.contrib.layers.fully_connected(feature, in_size, reuse=self.reuse, scope=scope,
                                                          activation_fn=self.activation_fn)
                    index += 1
                    features_h_lst.append(h)
                else:
                    features_h_lst.append(feature)

            h = tf.concat(features_h_lst, axis=-1)

            for layer in layers:
                if self.reuse:
                    scope = str(index)
                h = tf.contrib.layers.fully_connected(h, layer, reuse=self.reuse, scope=scope,
                                                      activation_fn=self.activation_fn)
                index += 1

            if self.reuse:
                scope = str(index)
            y = tf.contrib.layers.fully_connected(h, out, reuse=self.reuse, scope=scope, activation_fn=last_activation)

        return y

    def sgp(self, in_confidence_relation, in_confidence_entity, scope_name="rnn_cell"):
        """
        RNN stage - which get as an input a confidence of the predicates and objects and return an improved confidence of the predicates and the objects
        :return:
        :param in_confidence_relation: predicate confidence of the last stage in the RNN
        :param in_confidence_entity: object confidence of the last stage in the RNNS
        :param scope_name: rnn stage scope
        :return: improved predicates probabilties, improved predicate confidence,  improved object probabilites and improved object confidence
        """
        with tf.variable_scope(scope_name):

            # confidence to probes
            self.in_confidence_predicate_actual = in_confidence_relation
            predicate_probes = tf.nn.softmax(in_confidence_relation)
            self.predicate_probes = predicate_probes
            in_confidence_predicate_norm = tf.log(predicate_probes + tf.constant(1e-10))

            # FIXME check if can be removed
            in_confidence_entity = self.confidence_entity_ph
            self.in_confidence_object_actual = in_confidence_entity
            object_probes = tf.nn.softmax(in_confidence_entity)
            self.object_probes = object_probes
            in_confidence_object_norm = tf.log(object_probes + tf.constant(1e-10))

            # word embeddings
            # expand object word embed
            N = tf.slice(tf.shape(self.confidence_relation_ph), [0], [1], name="N")
            if self.gpi_type == "Linguistic":
                self.obj_prediction = tf.argmax(self.object_probes, axis=1)
                self.obj_prediction_val = tf.reduce_max(self.object_probes, axis=1)
                self.embed_objects = tf.gather(self.word_embed_entities_ph, self.obj_prediction)
                self.embed_objects = tf.transpose(
                    tf.multiply(tf.transpose(self.embed_objects), self.obj_prediction_val))
                in_extended_confidence_embed_shape = tf.concat((N, tf.shape(self.embed_objects)), 0)
                in_confidence_object_norm = tf.concat((self.embed_objects, in_confidence_object_norm), axis=1)

                self.pred_prediction = tf.argmax(self.predicate_probes[:, :, :self.nof_predicates - 1], axis=2)
                self.pred_prediction_val = tf.reduce_max(self.predicate_probes[:, :, :self.nof_predicates - 1], axis=2)
                self.embed_predicates = tf.gather(self.word_embed_relations_ph, tf.reshape(self.pred_prediction, [-1]))
                self.embed_predicates = tf.transpose(
                    tf.multiply(tf.transpose(self.embed_predicates), tf.reshape(self.pred_prediction_val, [-1])))
                self.embed_predicates = tf.reshape(self.embed_predicates, in_extended_confidence_embed_shape)
                in_confidence_predicate_norm = tf.concat((in_confidence_predicate_norm, self.embed_predicates), axis=2)

            # expand to NxN
            self.predicate_opposite = tf.transpose(in_confidence_predicate_norm, perm=[1, 0, 2])
            # expand object confidence    
            self.expand_object_confidence = tf.add(tf.zeros(self.extended_confidence_entity_shape),
                                                   in_confidence_object_norm,
                                                   name="expand_object_confidence")
            # expand subject confidence
            self.expand_subject_confidence = tf.transpose(self.expand_object_confidence, perm=[1, 0, 2],
                                                          name="expand_subject_confidence")

            ##
            # Node Neighbours
            # Subject features are self.expand_subject_confidence and self.bb_features[:3]
            # Object features are self.expand_object_confidence and self.bb_features[4:]
            # Pairwise featues are in_confidence_predicate_norm, self.predicate_opposite
            self.object_ngbrs = [self.expand_object_confidence, self.expand_subject_confidence,
                                 in_confidence_predicate_norm,
                                 self.predicate_opposite, self.bb_features]

            # Attention mechanism
            self.object_ngbrs_phi = self.nn(features=self.object_ngbrs, layers=[], out=500, scope_name="nn_phi")
            if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
                self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=[], out=500,
                                                   scope_name="nn_phi_atten")
                self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
                self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                          axis=1)

            elif self.gpi_type == "NeighbourAttention":
                self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=[], out=1,
                                                   scope_name="nn_phi_atten")
                self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
                self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                          axis=1)
            else:
                self.object_ngbrs_phi_all = tf.reduce_mean(self.object_ngbrs_phi, axis=1)

            ##
            # Nodes
            self.object_ngbrs2 = [in_confidence_object_norm, self.object_ngbrs_phi_all]
            self.object_ngbrs2_phi = self.nn(features=self.object_ngbrs2, layers=[], out=500, scope_name="nn_phi2")
            # Attention mechanism
            if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
                self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=[], out=500,
                                                    scope_name="nn_phi2_atten")
                self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
                self.object_ngbrs2_phi_all = tf.reduce_sum(
                    tf.multiply(self.object_ngbrs2_phi, self.object_ngbrs2_weights), axis=0)
            elif self.gpi_type == "NeighbourAttention":
                self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=[], out=1,
                                                    scope_name="nn_phi2_atten")
                self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
                self.object_ngbrs2_phi_all = tf.reduce_sum(
                    tf.multiply(self.object_ngbrs2_phi, self.object_ngbrs2_weights), axis=0)
            else:
                self.object_ngbrs2_phi_all = tf.reduce_mean(self.object_ngbrs2_phi, axis=0)

            expand_graph_shape = tf.concat((N, N, tf.shape(self.object_ngbrs2_phi_all)), 0)
            expand_graph = tf.add(tf.zeros(expand_graph_shape), self.object_ngbrs2_phi_all)

            ##
            # relation prediction
            # The input is object features, subject features, relation features and the representation of the graph
            self.expand_obj_ngbrs_phi_all = tf.add(tf.zeros_like(self.object_ngbrs_phi), self.object_ngbrs_phi_all)
            self.expand_sub_ngbrs_phi_all = tf.transpose(self.expand_obj_ngbrs_phi_all, perm=[1, 0, 2])
            self.predicate_all_features = [in_confidence_predicate_norm, self.predicate_opposite,
                                           self.expand_object_confidence,
                                           self.expand_subject_confidence, self.expand_sub_ngbrs_phi_all,
                                           self.expand_obj_ngbrs_phi_all, expand_graph, self.bb_features]

            pred_delta = self.nn(features=self.predicate_all_features, layers=self.layers, out=self.nof_predicates,
                                 scope_name="nn_pred")
            pred_forget_gate = self.nn(features=self.predicate_all_features, layers=[], out=1,
                                       scope_name="nn_pred_forgate", last_activation=tf.nn.sigmoid)
            out_confidence_predicate = pred_delta + pred_forget_gate * in_confidence_relation

            ##
            # entity prediction
            # The input is entity features, entity neighbour features and the representation of the graph
            if self.including_object:
                self.object_all_features = [in_confidence_object_norm, self.entity_bb_ph, expand_graph[0],
                                            self.object_ngbrs_phi_all]
                obj_delta = self.nn(features=self.object_all_features, layers=self.layers, out=self.nof_objects,
                                    scope_name="nn_obj")
                obj_forget_gate = self.nn(features=self.object_all_features, layers=[], out=self.nof_objects,
                                          scope_name="nn_obj_forgate", last_activation=tf.nn.sigmoid)
                out_confidence_object = obj_delta + obj_forget_gate * in_confidence_entity
            else:
                out_confidence_object = in_confidence_entity

            return out_confidence_predicate, out_confidence_object

    def module_loss(self, scope_name="loss"):
        """
        Set and minimize module loss
        :param lr: init learning rate
        :param lr_steps: steps to decay learning rate
        :param lr_decay: factor to decay the learning rate by
        :param scope_name: tensor flow scope name
        :return: loss and train step
        """
        with tf.variable_scope(scope_name):
            # reshape to batch like shape
            shaped_labels_predicate = tf.reshape(self.labels_relation_ph, (-1, self.nof_predicates))

            # predicate gt
            self.gt = tf.argmax(shaped_labels_predicate, axis=1)

            loss = 0

            for rnn_step in range(self.rnn_steps):

                shaped_confidence_predicate = tf.reshape(self.out_confidence_relation_lst[rnn_step],
                                                         (-1, self.nof_predicates))

                # set predicate loss
                self.predicate_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                                                                                 logits=shaped_confidence_predicate,
                                                                                 name="predicate_ce_loss")

                self.loss_predicate = self.predicate_ce_loss
                self.loss_predicate_weighted = tf.multiply(self.loss_predicate, self.labels_coeff_loss_ph)

                loss += tf.reduce_sum(self.loss_predicate_weighted)

                # set object loss
                if self.including_object:
                    self.object_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_entity_ph,
                                                                                  logits=self.out_confidence_entity_lst[
                                                                                      rnn_step],
                                                                                  name="object_ce_loss")

                    loss += self.lr_object_coeff * tf.reduce_sum(self.object_ce_loss)

            # reg             
            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars]) * self.reg_factor
            loss += lossL2

            # minimize
            # opt = tf.train.GradientDescentOptimizer(self.lr_ph)
            opt = tf.train.AdamOptimizer(self.lr_ph)
            # train_step = opt.minimize(loss)
            # opt = tf.train.MomentumOptimizer(self.lr_ph, 0.9, use_nesterov=True)
            # gradients = []
            # grad_placeholder = []
            gradients = opt.compute_gradients(loss)
            # create placeholder to minimize in a batch
            grad_placeholder = [(tf.placeholder("float", shape=grad[0].get_shape()), grad[1]) for grad in gradients]

            train_step = opt.apply_gradients(grad_placeholder)
        return loss, gradients, grad_placeholder, train_step

    def get_in_ph(self):
        """
        get input place holders
        """
        return self.confidence_relation_ph, self.confidence_entity_ph

    def get_output(self):
        """
        get module output
        """
        return self.out_relation_probes, self.out_entity_probes

    def get_labels_ph(self):
        """
        get module labels ph (used for train)
        """
        return self.labels_relation_ph, self.labels_entity_ph, self.labels_coeff_loss_ph

    def get_module_loss(self):
        """
        get module loss and train step
        """
        return self.loss, self.gradients, self.grad_placeholder, self.train_step
