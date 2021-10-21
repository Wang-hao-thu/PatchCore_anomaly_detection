export PYTHONPATH=/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD:$PYTHONPATH
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/Al_1/test_2.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/Al_3/test_2.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatchCore/PatchCore_anomaly_detection/datapath/0908-0924/test/test_50000_1083.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/test/test_2.lst'
imglist='/mnt/lustre/wanghao3/projects/DRAEM/data_path/ciwa_1/test/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/test/test_2.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/hongzao/test/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/hongzao/test/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/Cu_0/test_2.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/Cu_2/test_2.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/40_jier/test/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/test/test_50000_1083.lst'
#imglist='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/test/test_50000_1083.lst'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/Al_wai_1000/embedding.pickle'
embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/ciwa_1_1000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/merge_1000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/cu_wai_1000_500/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/cu_wai_2000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/merge_1000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatchCore/PatchCore_anomaly_detection/tools/embeddings/cu_nei_2000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/cu_nei_1000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/40_jier/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/al_nei_2000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/al_wai_2000/embedding.pickle'
load_size=256
input_size=224
out_dir=$1
defect_miss=$out_dir/miss.txt
partition=mediaa
split_num=16

if [ -d $out_dir ]
then
    rm -rf $out_dir
fi
mkdir $out_dir -p
num=1
in_file=$imglist

out_file_dir=${out_dir}/tmp_dir
mkdir $out_file_dir
out_file=${out_file_dir}/imglist
total_line=$(wc -l < "$in_file")
lines=$(echo $total_line/$split_num | bc -l)
line=${lines%.*}
line=$(expr $line + $num)
echo $line
split -l $line -d $in_file $out_file
#######
result_dir=$out_dir/defect
vis_dir=$out_dir/vis
mkdir $result_dir -p
for each_file in `ls ${out_file_dir}`
do
{
      sub_img_list=${out_file_dir}/${each_file}
          srun -p $partition -x SH-IDC1-10-5-30-[69,102] -n1 --gres=gpu:1 \
          python infer.py \
          --file_list=$sub_img_list \
          --save_file=$result_dir/$each_file.txt \
          --load_size=$load_size \
          --input_size=$input_size \
          --embedding_path=$embedding_path

} &
done
wait
echo done

result=$out_dir/results.all
cat $result_dir/* > $result
wc -l $result

python vis_result.py $result tmp/result_1.lst
