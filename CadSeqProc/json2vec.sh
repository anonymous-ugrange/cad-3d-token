python3 json2vec.py \
 --input_dir data/cad_json \
 --split_json data/train_val_test_split.json \
 --output_dir data/processed \
 --max_workers 8 \
 --padding \
 --deduplicate
