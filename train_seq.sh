python run_sequnce_labeling.py \
    --task_name=sequence_labeling \
    --do_train=true \
    --do_eval=true \
    --data_dir=openue/sequence_labeling/sequence_labeling_data/$1 \
    --vocab_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=9.0 \
    --output_dir=./output/sequnce_labeling_model/wwm/
