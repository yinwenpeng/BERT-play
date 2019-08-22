export GLUE_DIR=/home/wyin3/Datasets/glue_data
from pytorch_pretrained_bert.examples import run_classifier


python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /home/wyin3/Datasets/glue_data/MRPC/tmpoutput/
