from collections import namedtuple
import tensorflow as tf
import math
 
import layers as layers

class Model(object):
    
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None
        
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name : var for var in variables}

        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess =None):
        if not sess:
            raise AttributeError('Tensorflow session not provide')
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributedError('Tensorflow session not provided')
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print('Model restored from file: %s' % save_path)

class GeneralizedModel(Model):
    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        


SAGEInfo = namedtuple("SAGEInfo", ['layer_name', 'neigh_sampler', 'num_samples', 'output_dim'])

class SampleAndAggregate(GeneralizedModel):
    
    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean",
            model_size="small", identitity_dim=0, **kwargs):
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        else:
            raise Exception("Unknown aggregator:")

        #get info from palceholders ...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Identity dim and features have no positive values")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.conctant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat

        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def sample(self, inputs, layer_infos, batch_size=None, timestamp = None):
        
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]

        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            if timestamp is None:
                node = sampler((samples[k], layer_infos[t].num_samples))
            else:
                node = sampler((samples[k], layer_infos[t].num_samples, timestamp))
            samples.append(tf.reshape(node, [support_size * batch_size, ]))
            support_sizes.append(support_size)
        return samples, support_sizes


    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):

        if batch_size is None:
            batch_size = self.batch_size
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                if layer == len(num_samples) -1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x:x,
                                    dropout=self.placeholders['dropout'],
                                    name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                                    dropout=self.placeholders['dropout'],
                                    name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]

            next_hidden = []
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                            num_samples[len(num_samples) - hop -1],
                            dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop+1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators


    def _build_encoder(self, x):
        dim_mult = 2 if self.concat else 1 
        h = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1], 
                        dropout=self.placeholders['dropout'], act=tf.nn.relu)(x)
        mu = layers.Dense(dim_mult* self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])(h)
        log_sigma_squared = layers.Dense(dim_mult*self.dims[-1], dim_mult* self.dims[-1],
                        dropout=self.placeholders['dropout'])(h)

        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma

    def _build_decoder(self, z):
        dim_mult = 2 if self.concat else 1
        h = tf.contrib.keras.layers.Dense(dim_mult * self.dims[-1], activation='relu')(z)
        y_logit = tf.contrib.keras.layers.Dense(dim_mult * self.dims[-1])(h)
        y = tf.sigmoid(y_logit)
        return y_logit, y
        

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], tf.int64),
                [self.batch_size, 1])

        samples, support_sizes = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.hiddens, self.aggregators = self.aggregate(samples, [self.features], self.dims, num_samples,
                    support_sizes, concat=self.concat, model_size=self.model_size)
        
        self.hiddens = tf.nn.l2_normalize(self.hiddens, 1)
        self.mu, log_sigma_squared, sigma_squared, sigma = self._build_encoder(self.hiddens)
        self.z = tf.random_normal([self.dims], mean=self.mu, stddev=sigma)
        y_logit, self.y = self._build_decoder(self.z)



    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
        tf.summary.scalar('loss', self.loss)
        

    def build(self):
        self._build()
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op  = self.optimizer.apply_gradients(clipped_grad_and_vars)

    
