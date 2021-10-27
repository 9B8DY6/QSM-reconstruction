export CUDA_VISIBLE_DEVICES=0,1
python3 main.py \
--beta1 0.5 \
--beta2 0.999 \
--data_root './Data/phase/test' \
--experiment_name 'DIP_QSM' \
--gpu_ids '1' \
--init_gain 1 \
--init_type 'xavier' \
--lambda_tv 0.001 \
--lr 1e-3 \
--n_iters 500 \
--ngf 16 \
--save_path './Results'