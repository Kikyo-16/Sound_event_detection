sh scripts/gen_feature-2018.sh
sh scripts/gen_label-semi-2018.sh
CUDA_VISIBLE_DEVICES=3 python main.py -n DCASE2018-task4_semi -s sed_with_cATP-DF -t at_with_cATP-DF -u true -md train -g true
