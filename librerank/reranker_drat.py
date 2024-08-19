import itertools
import tensorflow as tf
import numpy as np
import heapq

from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell

def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))


def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)


class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                profile_num, max_norm=None):
        
        tf.reset_default_graph()

        self.graph = tf.Graph()
        with self.graph.as_default():

            with tf.name_scope('inputs'):
                self.itm_spar_ph = tf.placeholder(tf.int32, [None, max_time_len, itm_spar_num], name='item_spar')
                self.itm_dens_ph = tf.placeholder(tf.float32, [None, max_time_len, itm_dens_num], name='item_dens')

                self.usr_profile = tf.placeholder(tf.int32, [None, profile_num], name='usr_profile')
                self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
                self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')
                self.is_train = tf.placeholder(tf.bool, [], name='is_train')

                # lr
                self.lr = tf.placeholder(tf.float32, [])
                # reg lambda
                self.reg_lambda = tf.placeholder(tf.float32, [])
                # keep prob
                self.keep_prob = tf.placeholder(tf.float32, [])

                self.max_time_len = max_time_len
                self.hidden_size = hidden_size
                self.emb_dim = eb_dim
                self.itm_spar_num = itm_spar_num
                self.itm_dens_num = itm_dens_num
                self.profile_num = profile_num
                self.max_grad_norm = max_norm
                self.ft_num = itm_spar_num * eb_dim + itm_dens_num
                self.feature_size = feature_size

            with tf.name_scope('embedding'):
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim],
                                            initializer=tf.truncated_normal_initializer)
                self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)  # [b,l,n_s,d]

                self.item_seq = tf.concat(
                    [tf.reshape(self.itm_spar_emb, [-1, max_time_len, itm_spar_num * eb_dim]), self.itm_dens_ph], axis=-1) # [b,l,ft_num]

    def build_fc_net(self, inp, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred
    

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, dropout=0.9):
        with tf.variable_scope('pos_ff'):
            inp = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            l1 = tf.layers.conv1d(inp, d_inner_hid, 1, activation='relu')
            l2 = tf.layers.conv1d(l1, d_hid, 1)
            dp = tf.nn.dropout(l2, dropout, name='dp')
            dp = dp + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn2', training=self.is_train)
        return output

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw

    def transformer_layer(self, inp, d_model, d_inner_hid, dropout=0.9, n_head=2, scope='trans'):
        with tf.variable_scope(scope):
            pos_dim = inp.get_shape().as_list()[-1] 
            pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                    initializer=tf.truncated_normal_initializer)
            inp = inp + pos_mtx
            if pos_dim % 2:
                inp = tf.pad(inp, [[0, 0], [0, 0], [0, 1]])

            inp = self.multihead_attention(inp, inp, num_units=d_model, num_heads=n_head)
            inp = self.positionwise_feed_forward(inp, d_model, d_inner_hid, self.keep_prob)
            inp = tf.layers.dense(inp, d_model, activation=tf.nn.tanh, name='fc')

        return inp

    # ========== Loss & Optimization========== #

    def build_logloss(self, y_pred):
        # loss
        self.loss = tf.losses.log_loss(self.label_ph, y_pred)
        self.opt()

    def build_norm_logloss(self, y_pred):
        self.loss = - tf.reduce_sum(self.label_ph/(tf.reduce_sum(self.label_ph, axis=-1, keepdims=True) + 1e-8) * tf.log(y_pred))
        self.opt()

    def build_mseloss(self, y_pred):
        self.loss = tf.losses.mean_squared_error(self.label_ph, y_pred)
        self.opt()

    def build_attention_loss(self, y_pred):
        self.label_wt = attention_score(self.label_ph)
        self.pred_wt = attention_score(y_pred)
        self.loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
                # self.loss += self.reg_lambda * tf.norm(v, ord=1)

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
            })
            return loss

    def eval(self, batch_data, reg_lambda, keep_prob=1, no_print=True):
        with self.graph.as_default():
            pred, loss = self.sess.run([self.y_pred, self.loss], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: False,
            })
            return pred.reshape([-1, self.max_time_len]).tolist(), loss

    def set_sess(self, sess):
            self.sess = sess

    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=path)
            print('Save model:', path)

    def load(self, path):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('restore model:', ckpt.model_checkpoint_path)
    
# ========== DRAT Evaluator ========== #
class DRAT_evaluator(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                profile_num, eval_model, eval_loss, max_norm=None, d_model=64, n_head=1, hidden_layer_size=[128, 64, 32]):
        super(DRAT_evaluator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                            itm_spar_num, itm_dens_num, profile_num, max_norm)
        with self.graph.as_default():
            # GRU network
            if 'gru' in eval_model:
                gru_feature, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_seq,
                                                sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                scope='gru')  # [b,l,d] -> [b,l,hidden_size] -> hidden_size = 64
            # attention mechanism
            if 'attn' in eval_model:
                attn_feature = self.multihead_attention(self.item_seq, self.item_seq, num_units=d_model, num_heads=n_head) # [b,l,d_model] -> d_model = 64
            # bulid pair-network
            if 'pair' in eval_model:
                pair_input = self.extend_pair_feature(self.item_seq)  # item_seq -> [？, 10, ft_num * 2 + 1], 81 * 2 + 1 = 163
                pair_feature = self.build_pair_net(pair_input, d_model)   # [?, 10, 163] -> [?, 10, 64]
            
            if eval_model == 'gru_attn':
                inp = tf.concat([gru_feature, attn_feature], axis=-1)
            elif eval_model == 'gru_pair':
                inp = tf.concat([gru_feature, pair_feature], axis=-1)
            elif eval_model == 'attn_pair':
                inp = tf.concat([attn_feature, pair_feature], axis=-1)
            else:
                inp = tf.concat([gru_feature, attn_feature, pair_feature], axis=-1)
            # get reward
            self.y_pred = self.build_predict_function(inp, hidden_layer_size)

            # cross-entropy Loss
            if eval_loss == 'logloss':
                self.build_logloss(self.y_pred)
            # attention loss
            else:
                self.build_attention_loss(self.y_pred)

    def extend_pair_feature(self, inp):
        with tf.name_scope('pair_feature'):

            batch_size = tf.shape(self.item_seq)[0]

            pos = tf.reshape(tf.tile(tf.range(1, self.max_time_len + 1, 1), [batch_size]), [-1, self.max_time_len])  # [?,l]
            pos1 = tf.reshape(tf.tile(pos, [1, self.max_time_len]), [-1, self.max_time_len ** 2, 1])  # [?, l] -> [?,l^2, 1] -> [[1],[2],[3],[1],[2],[3],...]
            pos2 = tf.reshape(tf.tile(tf.reshape(pos, [-1, 1]), [1, self.max_time_len]), [-1, self.max_time_len ** 2, 1])# [?,l] -> [b,l^2, 1] -> [[1],[1],[1],[2],[2],[2],...]
            relative_pos = tf.cast(tf.subtract(pos1, pos2), dtype=tf.float32) * 0.1  # [?,l] -> [b,l^2, 1]

            # i1,i1,i1,i2,i2,i2,i3,i3,i3
            tile1 = tf.reshape(tf.tile(inp, [1, self.max_time_len, 1]), [-1, self.max_time_len ** 2, self.ft_num])  # [b,l^2,ft_num]
            # i1,i2,i3,i1,i2,i3,i1,i2,i3
            tile2 = tf.reshape(tf.tile(tf.reshape(inp, [-1, self.max_time_len * self.ft_num]), [1, self.max_time_len]), 
                                [-1, self.max_time_len ** 2, self.ft_num]) # [b,l^2,ft_num]

            pair_feature = tf.concat([tile1, tile2, relative_pos], -1)  # [b, l^2, 2ft_num + 1]
            
        return pair_feature

    def build_pair_net(self, pair_inp, d_model):

        inp = tf.layers.batch_normalization(inputs=pair_inp, name='pair_bn', training=self.is_train)
        pair_net = tf.layers.dense(inp, d_model, activation=None, name='pair_net')  # [b,l^2,d_model]
        pair_logits = tf.layers.dense(inp, 1, activation=None, name='pair_logits')  # [b,l^2,1]

        pair_net_dim = tf.reshape(pair_net, [-1, self.max_time_len, self.max_time_len, d_model])  # [b,l^2, d_model] -> [b,l,l,d_model]
        
        pair_logits = tf.reshape(pair_logits, [-1, self.max_time_len, self.max_time_len, 1])  # [b,l^2, 1] -> [b,l,l,1]

        pair_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32) # [b,l,l]
        pair_mask_reverse = 1.0 - pair_mask
        pair_mask_reverse = tf.reshape(pair_mask_reverse, [-1, 1, self.max_time_len, 1]) # [b*l,1,l,1]
        pair_logits = pair_logits + pair_mask_reverse * -80.0  # [b,l,l,1]

        pair_logit_dim = tf.nn.softmax(pair_logits, axis=2) # [b,l,l,1] -> [b,l,l]
        # point-wise
        merge_dim = tf.multiply(pair_net_dim, pair_logit_dim) # [b,l,l,d] * [b,l,l,1(broadcast)] -> [b,l,l,d]
        # pair net sum by one item (i1, i2; i1, i3; i1, i4;...)
        sum_pair_net = tf.reshape(tf.reduce_sum(merge_dim, axis=2), [-1, self.max_time_len, d_model])
        
        return sum_pair_net

    def build_predict_function(self, inp, layer, scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.elu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            score = seq_mask * score
            y_pred = score - tf.reduce_min(score, 1, keep_dims=True)
        return y_pred

    
    def predict(self, item_spar_fts, item_dens_fts, seq_len):
        with self.graph.as_default():
            ctr_probs = self.sess.run(self.y_pred, feed_dict={
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.keep_prob: 1.0,
                self.is_train: False})
            return ctr_probs
