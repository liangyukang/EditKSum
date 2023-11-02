#model=bert_bert12_cased
#model=bert_transformer_cased_Ggw
#model=bert_bert12_adaptor_cased_Ggw
model=transformer_transformer_cased

#concat=word
concat=sentence
#concat=concat_cons.txt
#concat=src
#concat=cons_src
#concat=empty_cons
#concat=empty

dataset=WikiSplit
y0=source

output_file=/data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/outputs/$dataset/$y0

CUDA_VISIBLE_DEVICES=1 fairseq-interactive \
	/data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/data-bin/bert-cased-$dataset \
	-s source -t target \
	--input /data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/concat/$dataset/concat.$y0 \
	--task translation_lev \
	--path /data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/checkpoints/$dataset/checkpoint_best.pt \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 5 \
	--print-step --retain-iter-history \
	--skip-invalid-size-inputs-valid-test \
    --max-source-positions 512 --max-target-positions 512 \
	--batch-size 128 \
	--buffer-size 128 \
	--has-target \
	--constrained-decoding \
	2>&1 > $output_file/output.txt 
#   --constrained-decoding \