import os
import copy
import math
from sklearn.metrics import log_loss, roc_auc_score
import time

from tqdm import tqdm
from librerank.utils import *
from librerank.reranker_drat import *
from librerank.rl_reranker_drat import *

# clip temperature
def generate_clip_temperature(x, max):
    if x > max // 5:
        return 0
    else:
        return 1

def generate_dynamic_temperature(x):
        return math.exp(-x)

def eval(model, data, l2_reg, batch_size, isrank, metric_scope, model_type, _print=False):
    preds = []
    labels = data[4]
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, loss = model.eval(data_batch, l2_reg)
        preds.extend(pred)
        losses.append(loss)

    loss = sum(losses) / len(losses)
    
    res = evaluate_multi(labels, preds, metric_scope, isrank, _print)

    eval_time = time.time() - t
    print("EVAL TIME: %.4fs" % eval_time)

    # return results
    return loss, res, eval_time


def train(train_file, test_file, feature_size, max_time_len, itm_spar_fnum, itm_dens_fnum, profile_num, params):
    tf.reset_default_graph()

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # For our model
    if params.model_type == 'DRAT_evaluator':
        model = DRAT_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_num, params.eval_model, params.eval_loss, max_norm=params.max_norm)
    elif params.model_type == 'DRAT_generator':
        model = PPO_DRAT_generator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_num, max_norm=params.max_norm, rep_num=params.rep_num)

        # load pre-trained evaluator
        evaluator = DRAT_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_num, params.eval_model, params.eval_loss, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.global_variables_initializer()) 
            evaluator.load(params.evaluator_path)
    else:
        print('No Such Model', params.model_type)
        exit(0)

    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.set_sess(sess)

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'map_l': [],
        'map_h': [],
        'ndcg_l': [],
        'clicks_l': [],
    }
    # store total eval results and inference time in each epoch
    total_eval_res = []
    total_eval_time = []
    
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params.timestamp, params.initial_ranker, params.model_type, params.batch_size,
                                params.lr, params.l2_reg, params.hidden_size, params.eb_dim, params.keep_prob, params.eval_model, params.eval_loss)
    if not os.path.exists('{}/logs_{}/{}'.format(parse.save_dir, data_set_name, max_time_len)):
        os.makedirs('{}/logs_{}/{}'.format(parse.save_dir, data_set_name, max_time_len))
    if not os.path.exists('{}/save_model_{}/{}/{}/'.format(parse.save_dir, data_set_name, max_time_len, model_name)):
        os.makedirs('{}/save_model_{}/{}/{}/'.format(parse.save_dir, data_set_name, max_time_len, model_name))
    save_path = '{}/save_model_{}/{}/{}/ckpt'.format(parse.save_dir, data_set_name, max_time_len, model_name)
    log_save_path = '{}/logs_{}/{}/{}.metrics'.format(parse.save_dir, data_set_name, max_time_len, model_name)
    
    result_path = '{}/{}/{}/{}_{}_res.txt'.format("./results", data_set_name, params.model_type, params.eval_model, params.eval_loss)

    train_losses_step = []

    # before training process
    step = 0
    vali_loss, res, eval_time = eval(model, test_file, params.l2_reg, params.batch_size, False, params.metric_scope, params.model_type)

    training_monitor['train_loss'].append(None)
    training_monitor['vali_loss'].append(None)
    training_monitor['map_l'].append(res[0][0])
    training_monitor['map_h'].append(res[0][-1])
    training_monitor['ndcg_l'].append(res[1][0])
    training_monitor['clicks_l'].append(res[2][0])

    # calculate initial results
    if not os.path.exists('{}/{}/{}'.format("./results", data_set_name, params.model_type)):
        os.makedirs('{}/{}/{}'.format("./results", data_set_name, params.model_type))
    
    print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)

    with open(result_path, 'a') as file:
        file.write('============ INTIAL RANKER RESULTS ============'.format(params.initial_ranker) + '\n')

        for i, s in enumerate(params.metric_scope):
            print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f " % (s, res[0][i], res[1][i], res[2][i]))
            file.write("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f " % (s, res[0][i], res[1][i], res[2][i]) + '\n')


    # define early_stop and iteration_counts
    early_stop = False
    iteration_counts = 0 

    data = train_file
    
    data_size = len(data[0])
    batch_num = data_size // params.batch_size
    eval_iter_num = (data_size // 5) // params.batch_size
    print('train', data_size, batch_num)

    # begin training process
    for epoch in tqdm(range(params.epoch_num)):
        
        # early_stop
        if early_stop:
            break

        for batch_no in range(batch_num):

            # early_stop
            if early_stop:
                break
            
            data_batch = get_aggregated_batch(data, batch_size=params.batch_size, batch_no=batch_no)  # [7(数据种类), 16(batch_size), 10(max_time_len)]

            if params.model_type == 'DRAT_generator':

                data_batch_copy = copy.deepcopy(data_batch)
                expert_data_batch, expert_act_idx_out = get_expert_aggregated_batch(data_batch_copy)

                data_batch = repeat_data(data_batch, params.rep_num)
                expert_data_batch = repeat_data(expert_data_batch, params.rep_num)
                
                act_idx_out, act_probs_one, rl_sparse_outputs, rl_dense_outputs, mask_arr, rerank_pred, expert_rerank_pred \
                    = model.predict(data_batch, expert_data_batch, expert_act_idx_out, params.keep_prob)

                rewards = evaluator.predict(rl_sparse_outputs, rl_dense_outputs, data_batch[6])
                
                if params.add_expert and params.temperature_weight == 'clip':
                    temperature = generate_clip_temperature(epoch, params.epoch_num)
                elif params.add_expert and params.temperature_weight == 'dynamic':
                    temperature = generate_dynamic_temperature(epoch)
                else:
                    temperature = 0
                
                loss, _ = model.train(data_batch, expert_data_batch, act_probs_one, act_idx_out,
                                    rewards, mask_arr, temperature, rerank_pred, expert_rerank_pred, params.c_entropy, params.lr, params.l2_reg, params.add_expert, params.keep_prob)

            elif params.model_type == 'DRAT_evaluator':
                loss = model.train(data_batch, params.lr, params.l2_reg, params.keep_prob)

            step += 1
            train_losses_step.append(loss)

            if step % eval_iter_num == 0:
                train_loss = sum(train_losses_step) / len(train_losses_step)
                training_monitor['train_loss'].append(train_loss)
                train_losses_step = []

                vali_loss, res, eval_time = eval(model, test_file, params.l2_reg, params.batch_size, True,
                                    params.metric_scope, params.model_type, False)

                training_monitor['train_loss'].append(train_loss)
                training_monitor['vali_loss'].append(vali_loss)
                training_monitor['map_l'].append(res[0][0])
                training_monitor['map_h'].append(res[0][-1])
                training_monitor['ndcg_l'].append(res[1][0])
                training_monitor['clicks_l'].append(res[2][0])

                print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (epoch, step, train_loss, vali_loss))
                for i, s in enumerate(params.metric_scope):
                    print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f " % (s, res[0][i], res[1][i], res[2][i]))

                if training_monitor['map_h'][-1] > max(training_monitor['map_h'][:-1]):
                    # save model
                    model.save(save_path)
                    pkl.dump(res[-1], open(log_save_path, 'wb'))
                    print('model saved')
                    # save results(for ad)
                    if params.data_set_name == 'ad':
                        total_eval_res.append(res)
                        total_eval_time.append(eval_time)

                if params.data_set_name == 'prm':
                    total_eval_res.append(res)
                    total_eval_time.append(eval_time)

                # count the nums of iteration
                iteration_counts += 1
                # early_stop（only vaild for prm）
                if params.data_set_name == 'prm':
                    if len(training_monitor['map_h']) > 20 and epoch >= (params.epoch_num // 4):
                        if (training_monitor['map_h'][-2] - training_monitor['map_h'][-1]) >= 0 and (
                                training_monitor['map_h'][-3] - training_monitor['map_h'][-2]) >= 0:
                            early_stop = True 

        # generate log
        if not os.path.exists('{}/logs_{}/{}/'.format(parse.save_dir, data_set_name, max_time_len)):
            os.makedirs('{}/logs_{}/{}/'.format(parse.save_dir, data_set_name, max_time_len))
        with open('{}/logs_{}/{}/{}.monitor.pkl'.format(parse.save_dir, data_set_name, max_time_len, model_name), 'wb') as f:
            pkl.dump(training_monitor, f)
    
    # calculate results by MAP_HIGH
    maph_total = [(res[0][-1], idx) for idx, res in enumerate(total_eval_res)]
    top_5_maph = sorted(maph_total, key=lambda x: x[0], reverse=True)[:5]

    with open(result_path, 'a') as file:
        file.write('============ METHOD: {} RERANK RESULTS ============'.format(params.model_type) + '\n')
        file.write('============ ADD EXPERT OR NOT: {} ============'.format(params.add_expert) + '\n')

        for i, scope in enumerate(params.metric_scope):
            file.write(' ====== SCOPE {} ======'.format(scope) + '\n')
            # store results
            res_scope = [[total_eval_res[top_idx][num][i] for _, top_idx in top_5_maph] for num in range(3)]
            file.write("@%d  MAP_MEAN: %.4f  NDCG_MEAN: %.4f  CLICKS_MEAN: %.4f " % 
                (scope, np.mean(res_scope[0]), np.mean(res_scope[1]), np.mean(res_scope[2])) + '\n')
            file.write("@%d  MAP_STD: %.4f  NDCG_STD: %.4f  CLICKS_STD: %.4f " % 
                (scope, np.std(res_scope[0]), np.std(res_scope[1]), np.std(res_scope[2])) + '\n')
        file.write("INFERENCE TIME_MEAN: %.4f" % np.mean([total_eval_time[top_idx] for _, top_idx in top_5_maph]) + '\n')
        file.write("ITERATION COUNTS: %d" % iteration_counts + '\n')

def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='./data/toy/', help='data dir')
    parser.add_argument('--model_type', default='DRAT_evaluator', choices=['DRAT_evaluator', 'DRAT_generator'],
                        type=str, help='algorithm name, including DRAT_evaluator, DRAT_generator')
    parser.add_argument('--data_set_name', default='ad', type=str, help='name of dataset, including ad and prm')
    parser.add_argument('--initial_ranker', default='lambdaMART', choices=['DNN', 'lambdaMART'], type=str, help='name of dataset, including DNN, lambdaMART')
    parser.add_argument('--epoch_num', default=30, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rep_num', default=5, type=int, help='samples repeat number')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_reg', default=1e-4, type=float, help='l2 loss scale')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='keep probability')
    parser.add_argument('--eb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size')
    parser.add_argument('--group_size', default=1, type=int, help='group size for GSF')
    parser.add_argument('--metric_scope', default=[1, 3, 5, 10], type=list, help='the scope of metrics')
    parser.add_argument('--max_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--c_entropy', default=0.001, type=float, help='entropy coefficient in loss')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='', help='evaluator ckpt dir')
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='./config/prm_setting.json', help='setting dir')
    # for evaluator ablation
    parser.add_argument('--eval_model', default='gru_attn_pair', choices = ['gru_attn', 'gru_pair', 'attn_pair', 'gru_attn_pair'], 
                        type=str, help='mode for selecting the models')
    parser.add_argument('--eval_loss', default='logloss', choices = ['logloss', 'attentionloss'], type=str, help='mode for selecting the loss')
    # for generator ablation
    parser.add_argument('--add_expert', action='store_true', help='ablation: using expert guidance')
    parser.add_argument('--temperature_weight', default='none', choices=['none', 'clip', 'dynamic'], type=str, help='the weight to adapt the similarity between predict and expert')

    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    parse = reranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)
    
    data_set_name = parse.data_set_name
    processed_dir = parse.data_dir
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = parse.max_time_len
    initial_ranker = parse.initial_ranker
    if data_set_name == 'prm' and parse.max_time_len > 30:
        max_time_len = 30
    print(parse)

    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
        stat['ft_num'], stat['profile_fnum'], stat['itm_spar_fnum'], stat['itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'], 'num of feature', num_ft,
        'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    # construct training files
    train_dir = os.path.join(processed_dir, initial_ranker + '.data.train')
    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        train_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.train'), max_time_len)
        pkl.dump(train_lists, open(train_dir, 'wb'))

    # construct test files if not exist
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.test'), max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    train(train_lists, test_lists, num_ft, max_time_len, itm_spar_fnum, itm_dens_fnum, profile_fnum, parse)




