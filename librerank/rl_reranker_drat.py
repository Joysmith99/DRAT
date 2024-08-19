import math
import os
import sys
import tensorflow as tf
import numpy as np
from librerank.reranker_drat import BaseModel

class DRAT_generator(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                profile_num, sample_val=0.2, max_norm=None, gamma=0.01, rep_num=1):
        super(DRAT_generator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                profile_num, max_norm)
        
        self.sample_val = sample_val
        self.gamma = gamma
        self.rep_num = rep_num
        
        with self.graph.as_default():
            self._build_graph()

    def _build_graph(self):

        with tf.variable_scope("input"):
            
            self.is_train = tf.placeholder(tf.bool, name="train_phase")
            self.use_expert = tf.placeholder(tf.bool, name="add_expert")
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")

            self.ex_itm_spar_ph = tf.placeholder(tf.int32, [None, self.max_time_len, self.itm_spar_num], name='ex_item_spar')
            self.ex_itm_dens_ph = tf.placeholder(tf.float32, [None, self.max_time_len, self.itm_dens_num], name='ex_item_dens')

            self.mask_in_raw = tf.placeholder(tf.float32, [None])
            self.expert_act_idx_out = tf.placeholder(dtype=tf.int32, shape=[None, self.max_time_len], name='expert_act_idx_out')

            self.item_input = self.item_seq
            self.item_label = self.label_ph
            self.item_size = self.max_time_len
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])
            self.enc_input = tf.reshape(self.item_input, [-1, self.item_size, self.ft_num])
            self.full_item_spar_fts = self.itm_spar_ph
            self.full_item_dens_fts = self.itm_dens_ph
            self.expert_item_spar_fts = self.ex_itm_spar_ph  
            self.expert_item_dens_fts = self.ex_itm_dens_ph

            self.ex_itm_spar_emb = tf.gather(self.emb_mtx, self.ex_itm_spar_ph)
            self.ex_input = tf.concat(
                [tf.reshape(self.ex_itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]), self.ex_itm_dens_ph], axis=-1)

            self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
            self.dec_input = tf.concat(
                [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]), self.itm_dens_ph], axis=-1)

        with tf.variable_scope("encoder"):
            enc_input_train = tf.reshape(tf.tile(self.enc_input, (1, self.max_time_len, 1)),
                                        [-1, self.item_size, self.ft_num])

            self.enc_output = tf.cond(self.is_train, lambda: enc_input_train, lambda: self.enc_input)

        with tf.variable_scope("decoder"):
            
            dec_input = tf.reshape(self.dec_input, [-1, self.max_time_len, self.ft_num])
            dec_output = self.transformer_layer(dec_input, d_model=self.ft_num, d_inner_hid=64, n_head=1)

            dec_output_train = tf.reshape(dec_output, [-1, 1, self.ft_num])
            dec_output_train_tile = tf.tile(dec_output_train, [1, self.item_size, 1])
            
            act_train = tf.concat([self.enc_output, dec_output_train_tile], axis = -1)

            self.act_logits_train = tf.reshape(self.build_dnn_net(act_train, [100, 50, 1], [tf.nn.elu, tf.nn.elu, None], "act_dnn"), [-1, self.item_size])
            self.act_probs_train_mask = tf.nn.softmax \
                (tf.add(tf.multiply(1. - self.mask_in, -1.0e9), self.act_logits_train))
            
            act_mask_tmp = tf.ones([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            self.random_val = tf.random_uniform([], 0, 1.0)

            mask_list = []
            act_idx_list = []
            act_probs_one_list = []
            act_probs_all_list = []
            next_dens_state_list = []
            next_spar_state_list = []
            scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)
            expert_scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            for k in range(self.max_time_len):

                mask_list.append(act_mask_tmp)
                mask_k = tf.reshape(tf.zeros_like(dec_output[:, k, :]), [-1, 1, self.ft_num])

                dec_output_mask_k = tf.concat([dec_output[:, :k, :], mask_k, dec_output[:, k+1:, :]], axis=1)
                
                act_input_mask = tf.reshape(dec_output_mask_k, [-1, self.item_size, self.ft_num])
                
                act_input = tf.concat([self.enc_output, act_input_mask], axis = -1)
                act_logits_pred = tf.reshape(self.build_dnn_net(act_input, [100, 50, 1], [tf.nn.elu, tf.nn.elu, None], "act_dnn"),
                                            [-1, self.item_size])

                act_probs_mask = tf.nn.softmax(tf.add(tf.multiply(1. - act_mask_tmp, -1.0e9), act_logits_pred))
                act_probs_mask_random = tf.nn.softmax(tf.add(tf.multiply(1. - act_mask_tmp, -1.0e9), act_mask_tmp))
                
                act_pred = tf.reshape(tf.multinomial(tf.log(act_probs_mask), num_samples=1), [-1])
                
                act_random = tf.reshape(tf.multinomial(tf.log(act_probs_mask_random), num_samples=1), [-1])

                act_idx_out = tf.cond(self.is_train, lambda: tf.cond(self.random_val < self.sample_val, 
                                                                    lambda: act_random, lambda: act_pred), 
                                                                    lambda: act_pred)

                tmp_range = tf.cast(tf.range(tf.shape(self.item_input)[0], dtype=tf.int32), tf.int64)
                
                idx_pair = tf.stack([tmp_range, act_idx_out], axis=1)
                
                idx_one_hot = tf.one_hot(act_idx_out, self.item_size)
                
                act_mask_tmp = act_mask_tmp - idx_one_hot

                next_full_spar_state = tf.gather_nd(self.full_item_spar_fts, idx_pair) 
                next_full_dens_state = tf.gather_nd(self.full_item_dens_fts, idx_pair)
                act_probs_one = tf.gather_nd(act_probs_mask, idx_pair)
                
                act_idx_list.append(act_idx_out)
                act_probs_one_list.append(act_probs_one)
                act_probs_all_list.append(act_probs_mask)
                next_spar_state_list.append(next_full_spar_state)
                next_dens_state_list.append(next_full_dens_state)
                
                scores_pred = scores_pred + tf.cast(idx_one_hot, dtype=tf.float32) * (1 / (1 + tf.log(tf.maximum(1.0, k + 1))))
                expert_scores_pred = self.item_label

            self.mask_arr = tf.stack(mask_list, axis=1)
            self.act_idx_out = tf.stack(act_idx_list, axis=1)
            self.act_probs_one = tf.stack(act_probs_one_list, axis=1)
            self.act_probs_all = tf.stack(act_probs_all_list, axis=1)
            self.next_spar_state_out = tf.reshape(tf.stack(next_spar_state_list, axis=1), [-1, self.full_item_spar_fts.shape[-1]])
            self.next_dens_state_out = tf.reshape(tf.stack(next_dens_state_list, axis=1), [-1, self.full_item_dens_fts.shape[-1]])

            self.rerank_predict = tf.identity(tf.reshape(scores_pred, [-1, self.max_time_len]), 'rerank_predict')
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            self.y_pred = self.rerank_predict * seq_mask
            
            self.expert_rerank_predict = tf.identity(tf.reshape(expert_scores_pred, [-1, self.max_time_len]), 'expert_rerank_predict')

        with tf.variable_scope("loss"):
            self._build_loss()

    def predict(self, batch_data, batch_expert_data, expert_act_idx_out, keep_prob, add_expert=False, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out, act_probs_one, next_state_spar_out, next_state_dens_out, mask_arr, rerank_predict, expert_rerank_predict = self.sess.run(
                [self.act_idx_out, self.act_probs_one, self.next_spar_state_out, self.next_dens_state_out, self.mask_arr,
                self.rerank_predict, self.expert_rerank_predict],
                feed_dict={
                            self.itm_spar_ph: batch_data[2], 
                            self.itm_dens_ph: batch_data[3],
                            self.ex_itm_spar_ph: batch_expert_data[2],
                            self.ex_itm_dens_ph: batch_expert_data[3],
                            self.expert_act_idx_out: expert_act_idx_out,
                            self.seq_length_ph: batch_data[6],
                            self.is_train: train_phase,
                            self.use_expert: add_expert,
                            self.sample_phase: sample_phase,
                            self.label_ph: batch_data[4],
                            self.keep_prob: keep_prob})
            return act_idx_out, act_probs_one, next_state_spar_out, next_state_dens_out, mask_arr, rerank_predict, expert_rerank_predict

    def eval(self, batch_data, reg_lambda, keep_prob=1, no_print=True):
        with self.graph.as_default():
            rerank_predict = self.sess.run(self.y_pred,
                feed_dict={
                    self.itm_spar_ph: batch_data[2],
                    self.itm_dens_ph: batch_data[3],
                    self.seq_length_ph: batch_data[6],
                    self.is_train: False,
                    self.use_expert: False,
                    self.sample_phase: False,
                    self.keep_prob: 1})
            return rerank_predict, 0

    def rank(self, batch_data, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out = self.sess.run(self.act_idx_out,
                                    feed_dict={
                                            self.itm_spar_ph: batch_data[2],
                                            self.itm_dens_ph: batch_data[3],
                                            self.is_train: train_phase,
                                            self.sample_phase: sample_phase})
            return act_idx_out

    def build_dnn_net(self, x, layer_nums, layer_acts, name="dnn"):
        input_ft = x
        assert len(layer_nums) == len(layer_acts)
        with tf.variable_scope(name):
            for i, layer_num in enumerate(layer_nums):
                input_ft = tf.contrib.layers.fully_connected(
                    inputs=input_ft,
                    num_outputs=layer_num,
                    scope='layer_%d' % i,
                    activation_fn=layer_acts[i],
                    reuse=tf.AUTO_REUSE)
            return input_ft
        
    def _build_loss(self):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def get_long_reward(self, rewards):
        long_reward = np.zeros(rewards.shape)
        val = 0
        for i in reversed(range(self.max_time_len)):
            long_reward[:, i] = self.gamma * val + rewards[:, i]
            val = long_reward[:, i]

        returns = long_reward[:, 0]
        return long_reward, returns

# Model-free gradient policy
class PPO_DRAT_generator(DRAT_generator):
    
    def _build_loss(self):
        self.clip_value = 0.1

        with tf.variable_scope("train_input"):
            self.old_act_prob = tf.placeholder(dtype=tf.float32, shape=[None], name='old_act_prob')
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')
            self.c_entropy = tf.placeholder(dtype=tf.float32, name='c_entropy')

        act_idx_one_hot = tf.one_hot(indices=self.actions, depth=self.item_size),  # [5*b,l,l]
        cur_act_prob = tf.reduce_sum(self.act_probs_train_mask * act_idx_one_hot, axis=-1)  # [5*b, l]
        ratios = tf.exp(tf.log(tf.clip_by_value(cur_act_prob, 1e-10, 1.0))
                        - tf.log(tf.clip_by_value(self.old_act_prob, 1e-10, 1.0)))
        
        self.ratio = ratios
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                        clip_value_max=1 + self.clip_value)
        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
        self.loss_clip = -tf.reduce_mean(loss_clip)
        self.mean_gaes = -tf.reduce_mean(self.gaes)

        # construct computation graph for loss of entropy bonus
        entropy = -tf.reduce_sum(self.act_probs_train_mask *
                                tf.log(tf.clip_by_value(self.act_probs_train_mask, 1e-10, 1.0)), axis=-1)
        self.entropy = tf.reduce_mean(entropy)  # mean of entropy

        # final loss
        self.loss = self.loss_clip - self.c_entropy * self.entropy

        self.g = tf.reduce_mean(self.returns)
        self.opt()


    def train(self, batch_data, expert_batch_data, old_act_prob, actions, rewards, act_mask, temperature, y_pred, expert_y_pred, 
            c_entropy, lr, reg_lambda, add_expert, keep_prop=0.8):

        with self.graph.as_default():

            # dynamic adjust rewards
            rewards = self.adapt_dynamic_weight(rewards, temperature, y_pred, expert_y_pred)  # [5*b,l]
            gaes, returns = self.get_gaes(rewards)
            
            _, total_loss, mean_return = self.sess.run(
                [self.train_step, self.loss, self.g],
                feed_dict={
                        self.itm_spar_ph: batch_data[2],
                        self.itm_dens_ph: batch_data[3],
                        self.label_ph: batch_data[4],
                        self.ex_itm_spar_ph: expert_batch_data[2],
                        self.ex_itm_dens_ph: expert_batch_data[3],
                        self.old_act_prob: old_act_prob.reshape([-1]),
                        self.actions: actions.reshape([-1]),
                        self.mask_in_raw: act_mask.reshape([-1]),
                        self.gaes: gaes,
                        self.returns: returns,
                        self.c_entropy: c_entropy,
                        self.reg_lambda: reg_lambda,
                        self.lr: lr,
                        self.keep_prob: keep_prop,
                        self.is_train: True,
                        self.use_expert: add_expert,
                        })
            return total_loss, mean_return

    def adapt_dynamic_weight(self, rewards, temperature, y_pred, expert_y_pred):
        # print(y_pred, expert_y_pred)
        # ===== cosine =====
        # sim = np.sum(y_pred * expert_y_pred, axis=1) / (np.linalg.norm(y_pred, axis=1) * np.linalg.norm(expert_y_pred, axis=1))  # [5*b]
        # rewards = (1 - temperature) * rewards + temperature * rewards * sim[:, np.newaxis]  # [5*b] -> [5*b,l]
        # ===== END ===== 
        # dot product
        dot_product = y_pred * expert_y_pred
        rewards = (1 - temperature) * rewards + temperature * rewards * dot_product
        # ===== euclidean distance ===== 
        # euclidean_distance = np.sqrt((y_pred - expert_y_pred) ** 2)
        # print(euclidean_distance)
        # rewards = (1 - temperature) * rewards + temperature * rewards * (1.0 - euclidean_distance)
        # ===== END =====
        return rewards

    def get_gaes(self, rewards):
        
        long_reward, returns = self.get_long_reward(rewards)
        gaes = np.reshape(long_reward,
                        [-1, self.rep_num, self.max_time_len])
        gaes_std = gaes.std(axis=1, keepdims=True)
        gaes_std = np.where(gaes_std == 0, 1, gaes_std)
        gaes = (gaes - gaes.mean(axis=1, keepdims=True)) / gaes_std

        return gaes.reshape([-1]), returns.reshape([-1])