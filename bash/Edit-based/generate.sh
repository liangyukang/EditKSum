dataset=WikiLarge
mode=EditKSum


output_file=/data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/outputs/$dataset/$mode

CUDA_VISIBLE_DEVICES=2 fairseq-generate \
	-s source -t target \
	/data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/data-bin/bert-cased-$dataset/delete \
	--task translation_lev \
	--path /data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/checkpoints/$dataset/$mode/checkpoint_best.pt \
    --tokenizer_dir /data/yukangliang/预训练模型/bert-base-cased \
	--cached_features_dir /data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/cached-examples/$dataset \
    --gen-subset test \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 5 \
	--print-step --retain-iter-history \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 512 --max-target-positions 512 \
	--batch-size 128 \
	--decoder_input target \
	--constrained-decoding \
	--kpe \
	2>&1 > $output_file/output.txt

#	--constrained-decoding \
#   --kpe \
#   --decoder_input keywords \