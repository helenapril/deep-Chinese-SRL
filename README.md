# deep-Chinese-SRL
semantic role labeling based on deep learning

# Train bilstm+CRF model：
python -u train_crf.py --cell highway --dropout_mode True --dropout_keep_prob 0.9 --batch_size 60 --num_epochs 180 --train_dir check_highway --embedding_dim 128 --mark_embedding_dim 128 --num_layers 4 --hidden_size 200

Note that, train_dir is the directory to preserve the model；


# Train bilstm+softmax model：
python -u train.py --cell highway --dropout_mode True --dropout_keep_prob 0.9 --batch_size 60 --num_epochs 180 --train_dir check_highway --embedding_dim 128 --mark_embedding_dim 128 --num_layers 4 



# add layer-norm to LSTM(Advanced model)：
python -u train_tmp.py --cell highway --dropout_mode True --dropout_keep_prob 0.9 --batch_size 60 --num_epochs 180 --train_dir check_highway_LN --embedding_dim 128 --mark_embedding_dim 128 --num_layers 4 --LN_mode True 


# inference using bilstm+CRF model

CUDA_VISIBLE_DEVICES=0 python -u test_crf.py --cell highway  --train_dir check_highway_1(the directory to preserve the re-trained model) --embedding_dim 128 --mark_embedding_dim 128 --num_layers 4 --hidden_size 200 --load_model model.ckpt-28215(a specific model)

Note that, the answer is preserved by data/cpbtest_answer.txt.
