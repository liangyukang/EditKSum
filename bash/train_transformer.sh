dataset=XLSum
mode=transformer

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/data-bin/Ggw \
    --arch transformer --share-decoder-input-output-embed \
    --ddp-backend=no_c10d \
    --task translation \
    --save-dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/checkpoint/$dataset/$mode \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4000 \
    --save-interval 1 \
    --max-update 141200  \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --no-epoch-checkpoints \