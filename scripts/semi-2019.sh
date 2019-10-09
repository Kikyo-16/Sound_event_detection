#sh scripts/gen_feature-2019.sh
#sh scripts/gen_label-2019.sh
CUDA_VISIBLE_DEVICES=3 python main.py -n DCASE2019-task4_semi -s sed_with_cATP-DF -t at_with_cATP-DF -u true -md train -g true
