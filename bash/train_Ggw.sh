CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train \
	../data-bin-bert-uncased-Ggw \
	--save-dir ../checkpoints_transformer_transformer_uncased_Ggw \
	--ddp-backend=no_c10d \
	--task translation_lev \
	--criterion nat_loss \
	--arch kpe_editor_transformer_with_adapter \
	--noise random_delete_shuffle \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr 0.0005 --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates 5000 \
	--warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--share-all-embeddings --no-share-discriminator \
	--dropout 0.3 --weight-decay 0.01 \
	--decoder-learned-pos --encoder-learned-pos \
	--apply-bert-init \
	--fixed-validation-seed 7 \
	--batch-size 64 \
    --save-interval 1 \
	--max-update 100000 \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 512 --max-target-positions 512 \
	--cached_features_dir ../cached_examples_bert_cased_510 \
	--tokenizer_dir /data/yukangliang/预训练模型/bert-base-uncased \
	--cache_dir /data/yukangliang/预训练模型/bert-base-uncased  \
	--decoder_cache_dir /data/yukangliang/预训练模型/bert-base-uncased-decoder \
	--num-workers 0 \
	--share-decoder-input-output-embed \
	--decoder_adapter_dimention 2048 \
	--encoder transformer \
	--decoder transformer \
	--decoder_input target \
	--update-freq 8 \
	--no-epoch-checkpoints \
	--keywords_gran token \
	--layers_num 6,6,6 \
	--fp16
#	--constraint \
#--update-freq 8 \
#--fp16 \	
#--layers_num 12,12,12 \
#--early-exit 12,12,12 \
#--kpe \	
#--finetune_position_embedding \
#--finetune-embeddings \
#--use_adapter_bert \
#--no-share-maskpredictor
