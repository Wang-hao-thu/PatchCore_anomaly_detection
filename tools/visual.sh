#img_list='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/hongzao/test/test.lst'
img_list='/mnt/lustre/wanghao3/projects/DRAEM/data_path/ciwa_1/test/ng_20.lst'
save_path=$1
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/al_wai_2000/embedding.pickle'
#embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/hongzao/embedding.pickle'
embedding_path='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/ciwa_1_1000/embedding.pickle'
load_size=256
input_size=224

srun -p mediaa --gres=gpu:1 \
        python visual_cls.py \
        --file_list=$img_list \
        --save_path=$save_path \
        --embedding_path=$embedding_path \
        --load_size=$load_size \
        --input_size=$input_size

