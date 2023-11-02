# model=transformer_transformer
# model=transformer_transformer
# model=bert_bert12_adaptor
models="transformer_transformer_cased bert_transformer_cased bert_bert12_adaptor_cased"
concats="empty_cons cons cons_src src"
device=0
for model in $models;
do 
	for concat in $concats;
	do
		output_file=../output_story/$model/$concat
		CUDA_VISIBLE_DEVICES=$device fairseq-interactive \
			-s source -t target \
			/data/yukangliang/实验/BertKpeEditorWithAdaptor/data-bin-bert-cased-510 \
			--input /data/yukangliang/实验/BertKpeEditorWithAdaptor/concat/concat_story/$concat \
			--task translation_lev \
			--path /data/yukangliang/实验/BertKpeEditorWithAdaptor/checkpoints_$model/checkpoint_best.pt \
			--iter-decode-max-iter 10 \
			--iter-decode-eos-penalty 0 \
			--beam 1 \
			--print-step --retain-iter-history \
			--skip-invalid-size-inputs-valid-test \
			--max-source-positions 512 --max-target-positions 512 \
			--batch-size 1 \
			--buffer-size 1 \
			--has-target \
			--constrained-decoding \
			2>&1 > $output_file/output.txt
	done
	device=device+1
#   --constrained-decoding \
done