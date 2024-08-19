# ========== AD ========== #
# ===== Evaluator ===== #
python run_reranker_dev.py --setting_path=./example/config/ad/ours_evaluator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss
# for ablation
python run_reranker_dev.py --setting_path=./example/config/ad/ours_evaluator_setting.json --eval_model=gru_pair --eval_loss=attentionloss
python run_reranker_dev.py --setting_path=./example/config/ad/ours_evaluator_setting.json --eval_model=attn_pair --eval_loss=attentionloss
python run_reranker_dev.py --setting_path=./example/config/ad/ours_evaluator_setting.json --eval_model=gru_attn --eval_loss=attentionloss

# ===== Generator ===== #
python run_reranker_dev.py --setting_path=./example/config/ad/ours_generator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss
# for ablation
python run_reranker_dev.py --setting_path=./example/config/ad/ours_generator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss --add_expert --temperature_weight=clip
python run_reranker_dev.py --setting_path=./example/config/ad/ours_generator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss --add_expert --temperature_weight=dynamic

# ========== PRM ========== #
# ===== Evaluator ===== #
python run_reranker_dev.py --setting_path=./example/config/prm/ours_evaluator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss
# ===== Generator ===== #
python run_reranker_dev.py --setting_path=./example/config/prm/ours_generator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss
python run_reranker_dev.py --setting_path=./example/config/ad/ours_generator_setting.json --eval_model=gru_attn_pair --eval_loss=attentionloss --add_expert

