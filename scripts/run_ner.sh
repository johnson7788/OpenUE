python setup.py install



python main.py --gpus "0," --max_epochs 1  \
    --data_class REDataset \
    --litmodel_class RELitModel \
    --model_class BertForNER \
    --task_name ner \
    --batch_size 16 \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 256 \
    --check_val_every_n_epoch 1 \
    --data_dir ./dataset/ske  \
    --overwrite_cache \
    --lr 3e-5 