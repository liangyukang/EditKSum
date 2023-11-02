output_file=output.txt

dataset=Ggw
mode=EditKSum
y0=speed

output_path=/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/outputs/$dataset/$mode/$y0
cd $output_path
grep ^H $output_file | cut -c 3- | sort -n | cut -f3- >  hypo_tmp.txt
python /data/yukangliang/数据集/CNNDM/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py hypo_tmp.txt hypo.txt
grep ^T $output_file | cut -c 3- | sort -n | cut -f2- >  target_tmp.txt
python /data/yukangliang/数据集/CNNDM/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py target_tmp.txt target.txt
grep ^S $output_file | cut -c 3- | sort -n | cut -f2- >  source_tmp.txt
python /data/yukangliang/数据集/CNNDM/cnn_dm_distill_by_bart/bert-cased-tokenizer/postprocess.py source_tmp.txt source.txt

# files2rouge --ignore_empty_summary target.txt hypo.txt -s rouge.txt

# python /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/outputs/CNNDM/evaluate_prf/evaluate_prf.py \
# $output_path \
# /data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/prepro-dataset/CNNDM/test_candidate.json
#python /data/yukangliang/实验/BertKpeEditorWithAdaptor/Edit-based/outputs/SARI.py source.txt target.txt hypo.txt 
#files2rouge target_tmp.txt hypo_tmp.txt -s rouge_tmp.txt