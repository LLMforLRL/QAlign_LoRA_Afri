PROJECT_PATH="$(pwd)"
MODEL_PATH=/home/leelab-alignfreeze2/QAlign/gemma-2-9b.gsm8kafri_all.finetune.metamath_all.finetune

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/evaluate/scripts/afrimgsm_test.py \
    --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 16 \
    # --lang_only Bengali Thai Swahili Japanese Chinese German French Russian Spanish English