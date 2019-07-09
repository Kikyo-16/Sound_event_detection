sh scripts/gen_feature-2018.sh
sh scripts/gen_label-2018.sh
CUDA_VISIBLE_DEVICES=3 python main.py -n DCASE2018-task4 -s sed_with_cATP-DF -u false -md train -g false
