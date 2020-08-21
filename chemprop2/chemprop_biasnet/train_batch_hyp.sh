data_path=$1
sep_test_path=$2

dataset_type="classification"
file_name="$(basename -- $1)"
protein_name="${file_name%.*}"
save_dir="models_hyp/mpnn/$protein_name"
config_path="configs/$protein_name.json"
num_classes=2
split_type="random"
ensemble_size=1
num_folds=3
gpu=2
log_frequency=100
metric="f1"
seed=1
python ../train.py --data_path $data_path \
	--separate_test_path $sep_test_path \
	--dataset_type $dataset_type \
        --save_dir $save_dir \
        --split_type $split_type \
	--config_path $config_path \
        --ensemble_size $ensemble_size \
        --multiclass_num_classes $num_classes \
        --num_folds $num_folds \
	--gpu $gpu \
        --quiet \
	--seed $seed \
        --metric $metric \
