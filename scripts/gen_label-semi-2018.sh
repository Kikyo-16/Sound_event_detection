python feature_extraction/gen_label.py -c data/text/weak-2018.csv -l data/label -d false -v false
python feature_extraction/gen_label.py -c data/text/test-2018.csv -l data/label -d true -v false
python feature_extraction/gen_label.py -c data/text/eval-2018.csv -l data/label -d true -v false
