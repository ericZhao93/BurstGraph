import tensorflow as tf

import graphsage as models
import layers as layers
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator


flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    
    def __init__(self, num_classes,
                placeholders, features, adj, degrees,
                layer_infos, concat=True, aggregator_type="mean",
                model_size="small", sigmoid_loss=False, identity_dim=0,
                    **kwargs):

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        else:
            raise Exception("Unknown aggregator:", self.aggregator_cls)

        self.inputs = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("identity feature dimension must be positive")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat  = concat
        self.num_classes = num_classes
        self.dims = [ (0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def build(self):
        samples, support_size = self.sample(self.inputs, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.hiddens, self.aggregators = self.aggregate(samples, [self.features], self.dims, num_samples,
                                support_size, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1

        self.hiddens = tf.nn.l2_normalize(self.hiddens, 1)

        # VAE
        self.mu, log_sigma_squared, sigma_squared, sigma = self._build_encoder(self.hiddens)
        self.z = tf.random_normal([dim_mult*self.dims[-1]], mean=self.mu, stddev=sigma)
        self.regular_vae = -0.5 * tf.reduce_sum(1 + log_sigma_squared - tf.square(self.mu) - sigma_squared, 1)
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                        dropout=self.placeholders['dropout'],
                        act=tf.nn.sigmoid)

        self.outputs = self.node_pred(self.z)

        self._loss()
        
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                    for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        # multi-class loss
        #logits = tf.math.log(self.outputs)
        #one_hot_labels = tf.one_hot(self.placeholders['labels'], depth=self.num_classes, dtype=tf.float32, axis=-1)
        #multi_loss = tf.einsum('bij,bj->bi', one_hot_labels, logits)
        #multi_loss =  - tf.einsum('bi,bi->b', multi_loss, self.label_masks)
        multi_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.placeholders['labels'])
        multi_loss = tf.einsum('bi,bi->b', multi_loss, self.placeholders['weights'])
        self.loss += tf.reduce_mean(multi_loss)
        self.loss += tf.reduce_mean(self.regular_vae)

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.outputs


class TwoChannelGraphsage(models.SampleAndAggregate):
    
    def __init__(self, num_classes,
                placeholders, features, adj, degrees,
                layer_infos, concat=True, aggregator_type="mean",
                model_size="small", sigmoid_loss=False, identity_dim=0,
                    **kwargs):

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        else:
            raise Exception("Unknown aggregator:", self.aggregator_cls)

        self.inputs = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        self.temperature = 1.0
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("identity feature dimension must be positive")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat  = concat
        self.num_classes = num_classes
        self.dims = [ (0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def _build_encoder(self, x):
        dim_mult  = 2 if self.concat else 1
        fc1 = layers.Dense(dim_mult*self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'], act=tf.nn.relu)
        fc2 = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])
        fc3 = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])
        h = fc1(x)
        mu = fc2(h)
        log_sigma_squared = fc3(h)
        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma, [fc1,fc2,fc3]

    def build(self):
        samples, support_size = self.sample(self.inputs, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.hiddens, self.aggregators = self.aggregate(samples, [self.features], self.dims, num_samples,
                                support_size, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1

        self.hiddens = tf.nn.l2_normalize(self.hiddens, 1)

        # Two Channel VAE
        self.mu_rec, log_sigma_squared, sigma_squared, sigma_rec, self.vars_vae_rec = self._build_encoder(self.hiddens)
        self.z_rec = tf.random_normal([dim_mult*self.dims[-1]], mean=self.mu_rec, stddev=sigma_rec)
        self.normal_rec = -0.5 * tf.reduce_sum(1 + log_sigma_squared - tf.square(self.mu_rec) - sigma_squared, 1)
        
        self.node_pred_rec = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                        dropout=self.placeholders['dropout'],
                        act=None)

        self.outputs_rec = self.node_pred_rec(self.z_rec)

        self.mu_abn, log_sigma_squared, sigma_squared, sigma_abn, self.vars_vae_abn = self._build_encoder(self.hiddens)
        self.z_abn = tf.random_normal([dim_mult*self.dims[-1]], mean=self.mu_abn, stddev=sigma_abn)
        self.normal_abn = -0.5 * tf.reduce_sum(1 + log_sigma_squared - tf.square(self.mu_abn) - sigma_squared, 1)
        
        u = tf.random_uniform(shape=(self.batch_size, dim_mult*self.dims[-1]), dtype=tf.float32)
        self.hidden_trans = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                                dropout=self.placeholders['dropout'], act=tf.nn.sigmoid)
        self.s = self.hidden_trans(self.hiddens)

        self.s_abn = tf.sigmoid((tf.log(self.s + 1e-20) + tf.log(u + 1e-20) - tf.log(1-u + 1e-20)) / self.temperature)
        self.bernoulli_abn = tf.reduce_sum(tf.log(self.s + 1e-20) + tf.log(1 - self.s + 1e-20) - 2 * tf.log(0.5), 1)

        self.r_abn = tf.multiply(self.z_abn, self.s_abn)
        
        self.node_pred_abn = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                dropout=self.placeholders['dropout'],
                                act=None)
        self.outputs_abn = self.node_pred_abn(self.r_abn)
        print(self.outputs_rec.get_shape())
        
        self.outputs = tf.reduce_max(tf.concat([tf.expand_dims(tf.nn.sigmoid(self.outputs_rec),-1) , tf.expand_dims(tf.nn.sigmoid(self.outputs_abn),-1) ], axis=-1), axis=-1)
        print(self.outputs.get_shape())
        self._loss()
       
        self.output_rec = tf.nn.sigmoid(self.outputs_rec)
        self.output_abn = tf.nn.sigmoid(self.outputs_abn)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                    for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        regular_weight = 0.3 
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred_rec.vars.values():
            self.loss += FLAGS.weight_decay * tf.reduce_sum(tf.abs(var))
        for var in self.node_pred_abn.vars.values():
            self.loss += FLAGS.weight_decay * tf.reduce_sum(tf.abs(var))
        for var in self.hidden_trans.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for variable in self.vars_vae_abn:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for variable in self.vars_vae_rec:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # multi-class loss
        #logits = tf.math.log(self.outputs)
        #one_hot_labels = tf.one_hot(self.placeholders['labels'], depth=self.num_classes, dtype=tf.float32, axis=-1)
        #multi_loss = tf.einsum('bij,bj->bi', one_hot_labels, logits)
        #multi_loss =  - tf.einsum('bi,bi->b', multi_loss, self.label_masks)
        multi_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs_rec, labels=self.placeholders['labels_rec'])
        multi_loss = tf.einsum('bi,bi->b', multi_loss, self.placeholders['weights_rec'])
        self.loss += tf.reduce_mean(multi_loss)
        multi_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs_abn, labels=self.placeholders['labels_abn'])
        multi_loss = tf.einsum('bi,bi->b', multi_loss, self.placeholders['weights_abn'])
        self.loss += tf.reduce_mean(multi_loss)

        self.loss += regular_weight * tf.reduce_mean(self.bernoulli_abn)
        self.loss += regular_weight * tf.reduce_mean(self.normal_rec)
        self.loss += regular_weight * tf.reduce_mean(self.normal_abn)

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.outputs

class SeqGraphsage(models.SampleAndAggregate):
    
    def __init__(self, num_classes,
                placeholders, features, adj, degrees,
                layer_infos, concat=True, aggregator_type="mean",
                model_size="small", sigmoid_loss=False, identity_dim=0,
                    num_steps=5, **kwargs):

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        else:
            raise Exception("Unknown aggregator:", self.aggregator_cls)

        self.inputs = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        self.temperature = 1.0
        self.num_steps = num_steps
        if identity_dim > 0:
            self.embeds = [tf.get_variable("node_embeddings_{:d}".format(idx), [adj.get_shape().as_list()[1], identity_dim]) for idx in range(num_steps)]
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("identity feature dimension must be positive")
            self.features = self.embeds
        else:
            self.features = [tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False, name="feature_{:d}".format(idx)) for idx in range(num_steps)]
            if not self.embeds is None:
                self.features = [tf.concat([self.embeds[idx], self.features[idx]], axis=1) for idx in range(num_steps)]
        self.degrees = degrees
        self.concat  = concat
        self.wei_pos = FLAGS.weight_value
        self.num_classes = num_classes
        self.dims = [ (0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def _build_encoder(self, inputs, hiddens, fc1=None, fc2=None, fc3=None):
        dim_mult  = 2 if self.concat else 1
        if fc1 is None:
            fc1 = layers.Dense(2*dim_mult*self.dims[-1], dim_mult* self.dims[-1],
                            dropout=self.placeholders['dropout'], act=tf.nn.relu)
        if fc2 is None:
            fc2 = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])
        if fc3 is None:
            fc3 = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])
        x = tf.concat([inputs, hiddens], axis=-1)
        h = fc1(x)
        mu = tf.nn.l2_normalize(fc2(h), 1)
        log_sigma_squared = tf.nn.l2_normalize(fc3(h), 1)
        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma, fc1,fc2, fc3


    def _build_prior(self, x, fc1=None, fc2=None, fc3=None):
        dim_mult = 2 if self.concat else 1
        hidden = fc1(x)
        mu = tf.nn.l2_normalize(fc2(hidden), 1)
        log_sigma_squared = tf.nn.l2_normalize(fc3(hidden), 1)
        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma


    def build(self):
        dim_mult = 2 if self.concat else 1
        self.outputs_rec, self.outputs_abn = [], []
        #print(self.placeholders["labels_abn"].get_shape().as_list())
        self.hidden_states = []
        for timestamp in range(self.num_steps):
            print("timestamp:", timestamp)
            samples, support_size = self.sample(self.inputs, self.layer_infos, timestamp=timestamp)
            num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
            if timestamp == 0:
                self.hiddens, self.aggregators = self.aggregate(samples, self.features[timestamp], self.dims, num_samples,
                                support_size, concat=self.concat, model_size=self.model_size)
            else:
                self.hiddens, _ = self.aggregate(samples, self.features[timestamp], self.dims, num_samples,
                                support_size, aggregators=self.aggregators, concat=self.concat,
                                model_size=self.model_size)
            
            self.hiddens = tf.nn.l2_normalize(self.hiddens, 1)

            # Two Channel VAE
            
            if timestamp == 0:
                self.hidden_states.append(tf.zeros_like(self.hiddens, dtype=tf.float32))
                self.prior_noli_rec = layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=tf.nn.relu)
                self.prior_mu_rec   = layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=None)
                self.prior_sigma_rec= layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=None)
                self.prior_noli_abn = layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=tf.nn.relu) 
                self.prior_mu_abn   = layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=None)
                self.prior_sigma_abn= layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=None)
            
            if timestamp == 0:
                self.mu_rec, log_sigma_squared_rec, sigma_squared_rec, sigma_rec, self.fc1_vae_rec, self.fc2_vae_rec, self.fc3_vae_rec = self._build_encoder(self.hiddens, self.hidden_states[-1])
            else:
                self.mu_rec, log_sigma_squared_rec, sigma_squared_rec, sigma_rec, fc1, fc2, fc3 = self._build_encoder(self.hiddens, self.hidden_states[-1], 
                                                                        fc1=self.fc1_vae_rec, fc2=self.fc2_vae_rec, fc3=self.fc3_vae_rec)
            self.z_rec = tf.random_normal([dim_mult* self.dims[-1]], mean=self.mu_rec, stddev=sigma_rec)
            self.sigma_rec = sigma_rec

            # KL divergence with prior distribution and approximate postier distribution
            if timestamp == 0:
                self.normal_rec = -0.5 * tf.reduce_mean(1 + log_sigma_squared_rec - tf.square(self.mu_rec) - sigma_squared_rec, 1)
            else:
                mu_pri, log_sigma_squared_pri, sigma_squared_pri, sigma_pri = self._build_prior(self.hidden_states[-1], fc1=self.prior_noli_rec, 
                                                            fc2=self.prior_mu_rec, fc3=self.prior_sigma_rec)
                sigma_squared_pri = sigma_squared_pri + 1e-10
                self.normal_rec += -0.5 * tf.reduce_mean(1 + log_sigma_squared_rec - log_sigma_squared_pri + tf.divide(sigma_squared_rec, sigma_squared_pri) - tf.divide(tf.square(self.mu_rec - mu_pri), sigma_squared_pri), 1)
                
            
            if timestamp == 0:
                self.node_pred_rec = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                        dropout=self.placeholders['dropout'],
                        act=None)

            outputs_rec = self.node_pred_rec(self.z_rec)
            
            if timestamp == 0:
                self.mu_abn, log_sigma_squared_abn, sigma_squared_abn, sigma_abn, self.fc1_vae_abn, self.fc2_vae_abn, self.fc3_vae_abn = self._build_encoder(self.hiddens, self.hidden_states[-1])
            else:
                self.mu_abn, log_sigma_squared_abn, sigma_squared_abn, sigma_abn, fc1, fc2, fc3 = self._build_encoder(self.hiddens, self.hidden_states[-1],
                                                                fc1=self.fc1_vae_rec, fc2=self.fc2_vae_rec, fc3=self.fc3_vae_rec)
            self.z_abn = tf.random_normal([dim_mult*self.dims[-1]], mean=self.mu_abn, stddev=sigma_abn)
            
            if timestamp == 0:
                self.normal_abn = -0.5 * tf.reduce_mean(1 + log_sigma_squared_abn - tf.square(self.mu_abn) - sigma_squared_abn, 1)
            else:
                mu_pri, log_sigma_squared_pri, sigma_squared_pri, sigma_pri = self._build_prior(self.hidden_states[-1], fc1=self.prior_noli_abn,
                                                                fc2=self.prior_mu_abn, fc3=self.prior_sigma_abn)
                sigma_pri = tf.math.reciprocal(sigma_pri+1e-10)
                sigma_trace = tf.multiply(sigma_pri, sigma_abn)
                mu_sigma = tf.multiply(mu_pri - self.mu_abn, sigma_pri)
                mu_sigma = tf.multiply(sigma_pri, mu_pri - self.mu_abn)
                self.normal_abn += -0.5 * tf.reduce_mean(1 + log_sigma_squared_abn - log_sigma_squared_pri + tf.divide(sigma_squared_abn, sigma_squared_pri) - tf.divide(tf.square(self.mu_abn - mu_pri), sigma_squared_pri), 1)
            
            self.sigma_abn = sigma_abn
            u = tf.random_uniform(shape=(self.batch_size, dim_mult*self.dims[-1]), dtype=tf.float32)
            if timestamp == 0:
                self.bernoulli_trans = layers.Dense(2*dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                                        dropout=self.placeholders['dropout'], act=tf.nn.sigmoid)
                self.bernoulli_prior = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                                        dropout=self.placeholders['dropout'], act=tf.nn.sigmoid)
            self.s = self.bernoulli_trans(tf.concat([self.hiddens, self.hidden_states[-1]], axis=-1))
            self.sp = self.bernoulli_prior(self.hidden_states[-1])
            self.s_abn = tf.sigmoid((tf.log(self.s + 1e-20) + tf.log(u + 1e-20) - tf.log(1-u + 1e-20)) / self.temperature)
            
            if timestamp == 0:
                self.bernoulli_abn = - 0.5 * tf.reduce_mean(tf.log(self.s + 1e-20) + tf.log(1 - self.s + 1e-20) - 2 * tf.log(0.5), 1)
            else:
                self.bernoulli_abn += - tf.reduce_mean(self.s*(tf.log(self.s + 1e-20) - tf.log(self.sp + 1e-20)) +(1-self.s) *(tf.log(1 - self.s + 1e-20) - tf.log(1 - self.sp + 1e-20)) , 1)
            self.r_abn = tf.multiply(self.z_abn, self.s_abn)
            if timestamp == 0:
                self.node_pred_abn = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                dropout=self.placeholders['dropout'],
                                act=None)

            outputs_abn = self.node_pred_abn(self.r_abn)
            self.outputs_rec.append(outputs_rec)
            self.outputs_abn.append(outputs_abn)
            
            if timestamp == 0:
                self.hidden_trans = layers.Dense(6*dim_mult*self.dims[-1], dim_mult*self.dims[-1],
                                            dropout=self.placeholders['dropout'],
                                            act=tf.nn.relu)

            next_hidden_state = self.hidden_trans(tf.concat([self.hidden_states[-1], self.mu_rec, self.sigma_rec, self.mu_abn, self.sigma_abn, self.s], axis=-1))
            self.hidden_states.append(tf.nn.l2_normalize(next_hidden_state,1))
        self._loss()
       
        self.output_rec = tf.nn.sigmoid(self.outputs_rec[-1])
        self.output_abn = tf.nn.sigmoid(self.outputs_abn[-1])
        self.outputs = tf.clip_by_value(self.output_abn, 1e-8, 1.0-1e-8)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None, var)
                                    for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        regular_weight = 1.0
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred_rec.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred_abn.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.hidden_trans.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        for variable in [self.fc1_vae_rec, self.fc2_vae_rec, self.fc3_vae_rec]:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for variable in [self.fc1_vae_abn, self.fc2_vae_abn, self.fc3_vae_abn]:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        for variable in [self.prior_noli_rec, self.prior_mu_rec, self.prior_sigma_rec]:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for variable in [self.prior_noli_abn, self.prior_mu_abn, self.prior_sigma_abn, self.bernoulli_trans, self.bernoulli_prior]:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        labels_rec = tf.unstack(self.placeholders['labels_rec'], axis=1)
        weights_rec = tf.unstack(self.placeholders['weights_rec'], axis=1)
        

        for idx in range(self.num_steps):
            self.loss += tf.reduce_mean(self._balanced_loss(self.outputs_rec[idx], labels_rec[idx], axis=1, wpos=self.wei_pos)) 
            self.loss += tf.reduce_mean(self._balanced_loss(self.outputs_rec[idx], labels_rec[idx], axis=0, wpos=self.wei_pos)) 

        labels_abn = tf.unstack(self.placeholders['labels_abn'], axis=1)
        weights_abn = tf.unstack(self.placeholders['weights_abn'], axis=1)
        for idx in range(self.num_steps):
            self.loss += tf.reduce_mean(self._balanced_loss(self.outputs_abn[idx], labels_abn[idx], axis=1, wpos=self.wei_pos*2)) #223
            self.loss += tf.reduce_mean(self._balanced_loss(self.outputs_abn[idx], labels_abn[idx], axis=0, wpos=self.wei_pos*2))  #118
        
        self.loss += tf.reduce_mean(self.bernoulli_abn)
        self.loss += tf.reduce_mean(self.normal_rec)
        self.loss += tf.reduce_mean(self.normal_abn)

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.outputs
    
    def _balanced_loss(self, logits, labels, axis=None, wpos=1.0):
        if axis is None:
            axis=-1
        pos_weight = tf.cast(tf.equal(labels, 1), tf.float32)
        neg_weight = 1 - pos_weight
        n_pos = tf.reduce_sum(pos_weight)
        n_neg = tf.reduce_sum(neg_weight)
        n_pos_divid = tf.reduce_sum(pos_weight, axis=axis)
        n_neg_divid = tf.reduce_sum(neg_weight, axis=axis)

        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        def has_pos():
            return tf.reduce_sum(ce_loss * pos_weight, axis=axis) / (n_pos_divid + 1e-5)
        def has_neg():
            return tf.reduce_sum(ce_loss * neg_weight, axis=axis) / (n_neg_divid + 1e-5)
        def is_zero():
            return tf.constant(0.0)
        pos_loss = tf.cond(n_pos > 0, has_pos, is_zero)
        neg_loss = tf.cond(n_neg > 0, has_neg, is_zero)
        return (pos_loss * wpos + neg_loss) / (wpos + 1.0)

class SeqTestGraphsage(models.SampleAndAggregate):
    
    def __init__(self, num_classes,
                placeholders, features, adj, degrees,
                layer_infos, concat=True, aggregator_type="mean",
                model_size="small", sigmoid_loss=False, identity_dim=0,
                    num_steps=5, **kwargs):

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        else:
            raise Exception("Unknown aggregator:", self.aggregator_cls)

        self.inputs = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        self.temperature = 1.0
        self.num_steps = num_steps
        if identity_dim > 0:
            self.embeds = [tf.get_variable("node_embeddings_{:d}".format(idx), [adj.get_shape().as_list()[1], identity_dim]) for idx in range(num_steps)]
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("identity feature dimension must be positive")
            self.features = self.embeds
        else:
            self.features = [tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False, name="feature_{:d}".format(idx)) for idx in range(num_steps)]
            if not self.embeds is None:
                self.features = [tf.concat([self.embeds[idx], self.features[idx]], axis=1) for idx in range(num_steps)]
        self.degrees = degrees
        self.concat  = concat
        self.num_classes = num_classes
        self.dims = [ (0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def _build_encoder(self, inputs, hiddens, fc1=None, fc2=None, fc3=None):
        dim_mult  = 2 if self.concat else 1
        if fc1 is None:
            fc1 = layers.Dense(2*dim_mult*self.dims[-1], dim_mult* self.dims[-1],
                            dropout=self.placeholders['dropout'], act=tf.nn.relu)
        if fc2 is None:
            fc2 = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])
        if fc3 is None:
            fc3 = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])
        x = tf.concat([inputs, hiddens], axis=-1)
        h = fc1(x)
        mu = fc2(h)
        log_sigma_squared = fc3(h)
        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma, fc1,fc2, fc3


    def _build_prior(self, x, fc1=None, fc2=None, fc3=None):
        dim_mult = 2 if self.concat else 1
        hidden = fc1(x)
        mu = fc2(hidden)
        log_sigma_squared = fc3(hidden)
        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma


    def build(self):
        dim_mult = 2 if self.concat else 1
        self.outputs_rec, self.outputs_abn = [], []
        #print(self.placeholders["labels_abn"].get_shape().as_list())
        self.hidden_states = []
        for timestamp in range(self.num_steps):
            print("timestamp:", timestamp)
            samples, support_size = self.sample(self.inputs, self.layer_infos, timestamp=timestamp)
            num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
            if timestamp == 0:
                self.hiddens, self.aggregators = self.aggregate(samples, self.features[timestamp], self.dims, num_samples,
                                support_size, concat=self.concat, model_size=self.model_size)
            else:
                self.hiddens, _ = self.aggregate(samples, self.features[timestamp], self.dims, num_samples,
                                support_size, aggregators=self.aggregators, concat=self.concat,
                                model_size=self.model_size)
            
            self.hiddens = tf.nn.l2_normalize(self.hiddens, 1)

            # Two Channel VAE
            
            if timestamp == 0:
                self.hidden_states.append(tf.ones_like(self.hiddens, dtype=tf.float32))
                self.prior_noli_rec = layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=tf.nn.relu)
                self.prior_mu_rec   = layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=None)
                self.prior_sigma_rec= layers.Dense(dim_mult * self.dims[-1], dim_mult * self.dims[-1],
                                    dropout=self.placeholders['dropout'], act=None)
            
            if timestamp == 0:
                self.mu_rec, log_sigma_squared_rec, sigma_squared_rec, sigma_rec, self.fc1_vae_rec, self.fc2_vae_rec, self.fc3_vae_rec = self._build_encoder(self.hiddens, self.hidden_states[-1])
            else:
                self.mu_rec, log_sigma_squared_rec, sigma_squared_rec, sigma_rec, fc1, fc2, fc3 = self._build_encoder(self.hiddens, self.hidden_states[-1], 
                                                                        fc1=self.fc1_vae_rec, fc2=self.fc2_vae_rec, fc3=self.fc3_vae_rec)
            self.z = tf.random_normal([dim_mult* self.dims[-1]], mean=self.mu_rec, stddev=sigma_rec)
            self.sigma_rec = sigma_rec

            # KL divergence with prior distribution and approximate postier distribution
            if timestamp == 0:
                self.normal_rec = -0.5 * tf.reduce_sum(1 + log_sigma_squared_rec - tf.square(self.mu_rec) - sigma_squared_rec, 1)
            else:
                mu_pri, log_sigma_squared_pri, sigma_squared_pri, sigma_pri = self._build_prior(self.hidden_states[-1], fc1=self.prior_noli_rec, 
                                                            fc2=self.prior_mu_rec, fc3=self.prior_sigma_rec)
                sigma_pri = tf.math.reciprocal(sigma_pri+1e-10)
                sigma_trace = tf.multiply(sigma_pri, sigma_rec)
                mu_sigma = tf.multiply(mu_pri - self.mu_rec, sigma_pri)
                mu_sigma = tf.multiply(sigma_pri, mu_pri - self.mu_rec)
                self.normal_rec += -0.5 * tf.reduce_sum(1 + log_sigma_squared_rec - log_sigma_squared_pri - sigma_trace - mu_sigma, 1)
                
            
            if timestamp == 0:
                self.node_pred_rec = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                        dropout=self.placeholders['dropout'],
                        act=None)

            outputs_rec = self.node_pred_rec(self.z)
           
            
            u = tf.random_uniform(shape=(self.batch_size, FLAGS.num_classes), dtype=tf.float32)
            if timestamp == 0:
                self.bernoulli_trans = layers.Dense(2*dim_mult* self.dims[-1], FLAGS.num_classes,
                                        dropout=self.placeholders['dropout'], act=tf.nn.sigmoid)
                self.node_pred_abn = layers.Dense(2 * dim_mult * self.dims[-1], FLAGS.num_classes,
                                        dropout=self.placeholders['dropout'], act=None)

            self.s = self.bernoulli_trans(tf.concat([self.z, self.hidden_states[-1]], axis=-1))
            self.s_abn = tf.sigmoid((tf.log(self.s + 1e-20) + tf.log(u + 1e-20) - tf.log(1-u + 1e-20)) / self.temperature)
            self.z_abn = self.node_pred_abn(tf.concat([self.z, self.hidden_states[-1]], axis=-1))
            self.r_abn = tf.multiply(self.z_abn, self.s_abn)
            
            outputs_abn = self.r_abn
            print(outputs_rec.get_shape())
            self.outputs_rec.append(outputs_rec)
            self.outputs_abn.append(outputs_abn)
            #self.outputs = tf.reduce_max(tf.concat([tf.expand_dims(tf.nn.sigmoid(self.outputs_rec),-1) , tf.expand_dims(tf.nn.sigmoid(self.outputs_abn),-1) ], axis=-1), axis=-1)
            
            # output next hidden states
            if timestamp == 0:
                self.hidden_trans = layers.Dense(3*dim_mult*self.dims[-1] + FLAGS.num_classes, dim_mult*self.dims[-1],
                                            dropout=self.placeholders['dropout'],
                                            act=tf.nn.relu)

            next_hidden_state = self.hidden_trans(tf.concat([self.hidden_states[-1], self.mu_rec, self.sigma_rec, self.s], axis=-1))
            self.hidden_states.append(tf.nn.l2_normalize(next_hidden_state,1))
        self._loss()
       
        self.output_rec = tf.nn.sigmoid(self.outputs_rec[-1])
        self.output_abn = tf.nn.sigmoid(self.outputs_abn[-1])
        self.outputs = tf.clip_by_value(self.output_abn, 1e-8, 1.0-1e-8)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                    for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        regular_weight = 0.1 
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred_rec.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred_abn.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.hidden_trans.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        for variable in [self.fc1_vae_rec, self.fc2_vae_rec, self.fc3_vae_rec]:
            for var in variable.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        
        labels_rec = tf.unstack(self.placeholders['labels_rec'], axis=1)
        weights_rec = tf.unstack(self.placeholders['weights_rec'], axis=1)
        for idx in range(self.num_steps):
            multi_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs_rec[idx], labels=labels_rec[idx])
            multi_loss = tf.einsum('bi,bi->b', multi_loss, weights_rec[idx])
            self.loss += tf.reduce_mean(multi_loss)
        labels_abn = tf.unstack(self.placeholders['labels_abn'], axis=1)
        weights_abn = tf.unstack(self.placeholders['weights_abn'], axis=1)
        for idx in range(self.num_steps):
            multi_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs_abn[idx], labels=labels_abn[idx])
            multi_loss = tf.einsum('bi,bi->b', multi_loss, weights_abn[idx])
            self.loss += tf.reduce_mean(multi_loss)

        #self.loss += regular_weight * tf.reduce_mean(self.bernoulli_abn)
        self.loss += regular_weight * tf.reduce_mean(self.normal_rec)
        #self.loss += regular_weight * tf.reduce_mean(self.normal_abn)

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.outputs
