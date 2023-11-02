dataset=Ggw
y0=speed
mode=Editor

output_path=/data/yukangliang/实验/LevT/outputs/Ggw-uncased/speed
# output_path=/data/yukangliang/实验/BertKpeEditorWithAdaptor/Summarization/outputs/$dataset/$mode/$y0
output_file=${output_path}/output.txt

cd $output_path
grep ^E $output_file | cut -c 3- | sort -n | cut -f 1 >  actions.txt
grep ^I $output_file | cut -c 3- | sort -n | cut -f 2- >  iterations.txt
python /data/yukangliang/实验/BertKpeEditorWithAdaptor/bash/iterative_step.py $output_path

