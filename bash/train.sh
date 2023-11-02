dataset=Ggw
mode=EditKSum

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
	/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/data-bin/$dataset \
	--save-dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/checkpoint/$dataset/$mode \
	--ddp-backend=no_c10d \
	--task translation_lev \
	--criterion nat_loss \
	--arch kpe_editor_transformer_with_adapter \
	--noise random_delete_shuffle \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr 0.0001 --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates 14120 \
	--warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--share-all-embeddings --no-share-discriminator \
	--dropout 0.3 --weight-decay 0.01 \
	--decoder-learned-pos --encoder-learned-pos \
	--apply-bert-init \
	--fixed-validation-seed 7 \
	--max-tokens 4000 \
    --save-interval 1 \
	--max-update 141200 \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 512 --max-target-positions 512 \
	--cached_features_dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/cached_examples/$dataset\
	--tokenizer_dir /data/yukangliang/预训练模型/bert-base-cased \
	--cache_dir /data/yukangliang/预训练模型/bert-base-cased  \
	--decoder_cache_dir /data/yukangliang/预训练模型/bert-base-cased-decoder \
	--num-workers 0 \
	--share-decoder-input-output-embed \
	--decoder_adapter_dimention 2048 \
	--decoder_input target \
	--update-freq 8 \
	--no-epoch-checkpoints \
	--keywords_gran token \
	--fp16 \
	--encoder_finetune_dir /data/yukangliang/实验/KPE/results/train_Ggw/checkpoints/bert2combi.kp20k.bert.epoch_2.checkpoint \
	--encoder bert_adaptor \
	--decoder bert_adaptor \
	--layers_num 12,12,12 \
	--early-exit 12,12,12 \
	--kpe \
	--use_adapter_bert \


## Ggw
# batch-size 8000
# update-1-epoch 689

## CNNDM
# batch-size 8
# update-1-epoch 1122

## XLSum
# batch-size 8000
# update-1-epoch 799


##参数说明：
# --encoder=['transformer','bert','bert_adaptor']
# --decoder=['transformer','bert_adaptor']
# --kpe 在encoder增加关键词抽取组件 配合--cachad_examples
# --use_adapter_bert 只训练adaptor

# --constraint keywords 作为decoder的输入，已放弃

## Adaptor------------------------------------------------
# --encoder bert_adaptor \
# --decoder bert_adaptor \
# --layers_num 12,12,12 \
# --early-exit 12,12,12 \
# --use_adapter_bert \
# Encoder model params: 146077440 (Encoder num. trained: 37767168)
# Decoder model params: 234283008 (Decoder num. trained: 104294400)

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

## Finetune EditKSum -----------------------------------------
#--encoder_finetune_dir
#--encoder bert_adaptor_kpe \
#--decoder bert_adaptor \
#--layers_num 12,12,12 \
#--early-exit 12,12,12 \
#--kpe \
#--use_adapter_bert \
#--frozen [kpe,encoder]

# frozen False 
# Encoder model params: 146472707 (Encoder num. trained: 38162435)
# Decoder model params: 234283008 (Decoder num. trained: 104294400)

# frozen kpe  kpe 395262
# num. Encoder model params: 146472707 (Encoder num. trained: 37767168)
# num. Decoder model params: 234283008 (Decoder num. trained: 104294400)
##------------------------------------------------------------------------

## Editor ---------------------------------------------------
#--encoder transformer \
#--decoder transformer \

# Encoder model params: 55746816 (Encoder num. trained: 55746816)
# Decoder model params: 117590784 (Decoder num. trained: 117590784)
#------------------------------------------------------------

## EndtoEnd----------------------------------------------------
#--encoder bert
#--decoder transformer
#--kpe \	
# Encoder model params: 108705539 (Encoder num. trained: 108705539)
# Decoder model params: 117590784 (Decoder num. trained: 117590784)
#---------------------------------------------------------------------

## Finetune EndtoEnd-----------------------------------------------------------
# --encoder bert_kpe
# --decoder transformer
# --kpe \
# --encoder_finetune_dir /data/yukangliang/实验/KPE/results/train_CNNDM/checkpoints/bert2joint.kp20k.bert.epoch_2.checkpoint
# --frozen [kpe,encoder] kpe冻结kpe参数，encoder冻结encoder所有参数

# frozen False 
# Encoder model params: 108705539 (Encoder num. trained: 108705539)
# Decoder model params: 117590784 (Decoder num. trained: 117590784)

# frozen kpe
# Encoder model params: 108705539 (Encoder num. trained: 108310272)
# Decoder model params: 117590784 (Decoder num. trained: 117590784)
## ---------------------------------------------------------------------------

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
