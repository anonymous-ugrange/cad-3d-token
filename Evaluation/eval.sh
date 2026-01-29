# INTERVAL=600
# EXP_DIR="/data2/yijia/projects/Text2CAD/deepcad_log/ours_dec6_bs512_p0"

# for i in $(seq 620 20 1000); do
#     FILE="${EXP_DIR}/test/epoch_${i}/output.pkl"
#     OUT_DIR="${EXP_DIR}/test/epoch_${i}/indicator"

#     while [ ! -f "$FILE" ]; do
#         echo "file for epoch ${i} not exist, sleep ${INTERVAL}s"
#         sleep ${INTERVAL}
#     done

#     echo "processing file for epoch ${i}..."
#     python eval_seq.py --input_path ${FILE} --output_dir ${OUT_DIR}

# done

#!/bin/bash

DIR="/data2/yijia/others/complex/deepcad-lfa"

for file in "$DIR"/*; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        base="${file##*/}"
        name="${base%.*}"
        
        python eval_seq.py --input_path ${file} --output_dir "${DIR}/${name}"
    fi
done
