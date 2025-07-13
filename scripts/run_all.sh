# Model
models=("Qwen2.5-1.5B-Instruct" "Llama-3.2-1B-Instruct")
# Dataset
datasets=("2wikimultihopqa" "hotpotqa" "iirc" "strategyqa")

# Method
methods=("DRAGIN" "FL-RAG" "FLARE" "FS-RAG" "SR-RAG" "wo-RAG")

# Traverse all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            config_path="configs/${model}/${dataset}/${method}.json"

            echo "============================="
            echo "▶ Processing config: $config_path"
            echo "============================="

            CUDA_VISIBLE_DEVICES=0 python src/main.py -c "$config_path"

            echo "============================="
            echo "▶ Evaluating: result/${model}/${dataset}/${method}/0"
            echo "============================="

            python src/evaluate.py --dir "result/${model}/${dataset}/${method}/0"

            echo
        done
    done
done
