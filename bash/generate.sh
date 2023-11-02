return_mode=shers
return_num=1.0

dataset=CNNDM
#y0=${return_mode}_$return_num
y0=speed
mode=EndtoEnd


output_file=/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/outputs/$dataset/$mode/$y0

CUDA_VISIBLE_DEVICES=5 fairseq-generate \
	-s source -t target \
	/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/data-bin/$dataset \
	--task translation_lev \
	--path /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/checkpoint/$dataset/$mode/checkpoint_best.pt \
    --tokenizer_dir /data/yukangliang/预训练模型/bert-base-cased \
	--cached_features_dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/cached_examples/$dataset \
    --gen-subset test \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 \
	--print-step --retain-iter-history \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 512 --max-target-positions 512 \
	--batch-size 1 \
	--decoder_input target \
	--constrained-decoding \
  	--kpe \
	--return_num $return_num \
	--return_mode $return_mode \
	--constraint_file $output_file \
	2>&1 > $output_file/output.txt

# EditKSum -------------------------------------
#	--constrained-decoding \
#   --kpe \
# 	--return_num $return_num \
# 	--return_mode $return_mode \
# 	--constraint_file $output_file \
# ----------------------------------------------

# Editor ---------------------------------------
# 


#   --decoder_input keywords \ #已放弃 decoder的输入