dataset=WikiLarge
model=EditKSum

CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train \
	/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/data-bin/XLSum/510_dataset \
	--save-dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/checkpoint/XLSum/Editor \
	--ddp-backend=no_c10d \
	--task translation_lev \
	--criterion nat_loss \
	--arch kpe_editor_transformer_with_adapter \
	--noise random_delete_shuffle \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr 0.0005 --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates 7790 \
	--warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--share-all-embeddings --no-share-discriminator \
	--dropout 0.3 --weight-decay 0.01 \
	--decoder-learned-pos --encoder-learned-pos \
	--apply-bert-init \
	--fixed-validation-seed 7 \
	--max-tokens 8000 \
    --save-interval 1 \
	--max-update 155800 \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 512 --max-target-positions 512 \
	--cached_features_dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/cached-examples/$dataset \
	--tokenizer_dir /data/yukangliang/预训练模型/bert-base-cased \
	--cache_dir /data/yukangliang/预训练模型/bert-base-cased  \
	--decoder_cache_dir /data/yukangliang/预训练模型/bert-base-cased-decoder \
	--num-workers 0 \
	--share-decoder-input-output-embed \
	--decoder_adapter_dimention 2048 \
	--encoder transformer \
	--decoder transformer \
	--decoder_input target \
	--update-freq 8 \
	--no-epoch-checkpoints \
	--keywords_gran token \
	--fp16 \

##参数说明：
# --encoder=['transformer','bert','bert_adaptor']
# --decoder=['transformer','bert_adaptor']
# --kpe 在encoder增加关键词抽取组件 配合--cachad_examples
# --use_adapter_bert 只训练adaptor

# --constraint keywords 作为decoder的输入，已放弃


## EditKSum ----------------------------------------------
#--encoder bert_adaptor \
#--decoder bert_adaptor \
#--layers_num 12,12,12 \
#--early-exit 12,12,12 \
#--kpe \	
#--use_adapter_bert \

# Encoder model params: 146472707 (Encoder num. trained: 38162435)
# Decoder model params: 234283008 (Decoder num. trained: 104294400)
#----------------------------------------------------------

## Editor ---------------------------------------------------
#--encoder transformer \
#--decoder transformer \

# Encoder model params: 55746816 (Encoder num. trained: 55746816)
# Decoder model params: 117590784 (Decoder num. trained: 117590784)
#------------------------------------------------------------

#--constraint \
#--update-freq 8 \
#--fp16 \	
#--layers_num 12,12,12 \
#--early-exit 12,12,12 \
#--kpe \	
#--use_adapter_bert \


#--finetune_position_embedding \
#--finetune-embeddings \

#--no-share-maskpredictor
