output_file=output.txt
#models="transformer_transformer_cased bert_transformer_cased bert_bert12_adaptor_cased"
models="transformer_transformer_cased"
concats="word sentence"
for model in $models;
do 
	for concat in $concats;
	do
        cd /data/yukangliang/实验/BertKpeEditorWithAdaptor/output_story/$model/$concat
        grep ^H $output_file | cut -c 3- | sort -n | cut -f3- >  hypo_tmp.txt
        python /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py hypo_tmp.txt hypo.txt
        grep ^T $output_file | cut -c 3- | sort -n | cut -f2- >  target_tmp.txt
        python /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py target_tmp.txt target.txt
        grep ^S $output_file | cut -c 3- | sort -n | cut -f2- >  source_tmp.txt
        python /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py target_tmp.txt source.txt

        grep ^O $output_file | cut -c 3- | sort -n | cut -f2- >  nums_tmp.txt 
        python /data/yukangliang/实验/BertKpeEditorWithAdaptor/bash/avg_op_num.py nums_tmp.txt num.txt

        files2rouge --ignore_empty_summary target.txt hypo.txt -s rouge.txt
	done
done
