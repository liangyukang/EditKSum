output_file=output.txt

cd /data/yukangliang/实验/fairseq-editor-main/outputs/transformer_transformer_bert_cased_510
grep ^H $output_file | cut -c 3- | sort -n | cut -f3- >  hypo_tmp.txt
python /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py hypo_tmp.txt hypo.txt
#grep ^T $output_file | cut -c 3- | sort -n | cut -f2- >  target_tmp.txt
python /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/510_dataset/test.target target.txt
grep ^S $output_file | cut -c 3- | sort -n | cut -f2- >  source_tmp.txt

python /data/yukangliang/数据集/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py source_tmp.txt source.txt

files2rouge --ignore_empty_summary target.txt hypo.txt -s rouge.txt
#files2rouge target_tmp.txt hypo_tmp.txt -s rouge_tmp.txt