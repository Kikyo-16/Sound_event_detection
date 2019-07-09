PROCESS_NUM=20
python feature_extraction/gen_feature.py -l data/text/semi-2019.lst -w data/wav/ -f data/feature -c feature_extraction/feature.cfg -p $PROCESS_NUM
python feature_extraction/gen_feature.py -l data/text/test-2019.lst -w data/wav/ -f data/feature -c feature_extraction/feature.cfg -p $PROCESS_NUM
python feature_extraction/gen_feature.py -l data/text/eval-2019.lst -w data/wav/ -f data/feature -c feature_extraction/feature.cfg -p $PROCESS_NUM
