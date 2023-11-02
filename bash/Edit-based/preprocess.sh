src=source
tgt=target

dataset=XLSum


text=/data/yukangliang/数据集/XLSum/raw/bert-cased-tokenize/510_dataset
output_dir=/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/data-bin/XLSum/510_dataset

fairseq-preprocess \
--source-lang ${src} --target-lang ${tgt} \
--task translation_lev \
--trainpref $text/train --validpref $text/valid --testpref $text/test \
--destdir ${output_dir} --workers 60 \
--joined-dictionary \
--srcdict /data/yukangliang/预训练模型/bert-base-cased/vocab_.txt \
--padding-factor 0