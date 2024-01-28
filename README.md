# SubSelNet
Code for [Efficient Data Subset Selection to Generalize Training Across Models: Transductive and Inductive Networks](https://openreview.net/pdf?id=q3fCWoC9l0) by Jain et al., NeurIPS 2023
> Existing subset selection methods for efficient learning predominantly employ discrete combinatorial and model-specific approaches, which lack generalizability— for each new model, the algorithm has to be executed from the beginning. Therefore, for an unseen architecture, one cannot use the subset chosen for a different model. In this work, we propose SubSelNet, a non-adaptive subset selection framework, which tackles these problems. Here, we first introduce an attention-based neural gadget that leverages the graph structure of architectures and acts as a surrogate to trained deep neural networks for quick model prediction. Then, we use these predictions to build subset samplers. This naturally provides us two variants of SubSelNet. The first variant is transductive (called Transductive-SubSelNet), which computes the subset separately for each model by solving a small optimization problem. Such an optimization is still super fast, thanks to the replacement of explicit model training by the model approximator. The second variant is inductive (called Inductive-SubSelNet), which computes the subset using a trained subset selector, without any optimization. Our experiments show that our model outperforms several methods across several real datasets.

## Nasbench Architectures
Install [nasbench](https://github.com/google-research/nasbench) and download [nasbench_only108.tfrecord](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord).

- Extract information from the architectures and store them separately
```bash
usage: jsonify.py [-h] [--nasbench_path NASBENCH_PATH] [--out_folder OUT_FOLDER]

optional arguments:
  --nasbench_path NASBENCH_PATH
                        Path to the nasbench record file
  --out_folder OUT_FOLDER
                        Path to the directory to save the json files
```

- Create train-test splits for the architecture space
```bash
usage: sampler.py [-h] [--nasbench_path NASBENCH_PATH]
                  [--out_folder OUT_FOLDER] [--num_train NUM_TRAIN]
                  [--num_test NUM_TEST] [--seed SEED]

optional arguments:
  --nasbench_path NASBENCH_PATH
                        Path to the nasbench record file
  --out_folder OUT_FOLDER
                        Path to the directory to save the json files
  --num_train NUM_TRAIN
                        Number of training architectures
  --num_test NUM_TEST   Number of testing architectures
  --seed SEED           Seed for RNG (default: 0)
```

## Architecture Embeddings
We train the GNN in a GVAE fashion over the architecture space
- To train the GNN
```bash
usage: train_emb.py [-h] [--nasbench_path NASBENCH_PATH]
                    [--data_folder DATA_FOLDER]
                    [--checkpoint_folder CHECKPOINT_FOLDER]
                    [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--seed SEED]
                    [--dropout DROPOUT] [--input_dim INPUT_DIM]
                    [--hidden_dim HIDDEN_DIM] [--output_dim OUTPUT_DIM]
                    [--num_rec NUM_REC] [--num_layers NUM_LAYERS]

optional arguments:
  --nasbench_path NASBENCH_PATH
                        Path to the nasbench record file
  --data_folder DATA_FOLDER
                        Path to the data.json folder
  --checkpoint_folder CHECKPOINT_FOLDER
                        Path to the checkpoint save folder
  --batch_size BATCH_SIZE
                        Training batch size (default: 32)
  --epochs EPOCHS       Number of epochs (default: 10)
  --seed SEED           Seed for RNG (default: 0)
  --dropout DROPOUT     Dropout (default: 0.3)
  --input_dim INPUT_DIM
                        GNN input dim (default: 5)
  --hidden_dim HIDDEN_DIM
                        GNN hidden dim (default: 128)
  --output_dim OUTPUT_DIM
                        GNN output dim (default: 16)
  --num_rec NUM_REC     GNN rec (default: 5)
  --num_layers NUM_LAYERS
                        GNN layers (default: 2)
```
- To generate embeddings over the train-test split
```bash
usage: gen_emb.py [-h] [--json_folder JSON_FOLDER] [--model_path MODEL_PATH]
                  [--out_folder OUT_FOLDER] [--dropout DROPOUT]
                  [--input_dim INPUT_DIM] [--hidden_dim HIDDEN_DIM]
                  [--output_dim OUTPUT_DIM] [--num_rec NUM_REC]
                  [--num_layers NUM_LAYERS] [--train]

optional arguments:
  --json_folder JSON_FOLDER
                        Path to the folder containing architecture json files
  --model_path MODEL_PATH
                        Path to the checkpoint file
  --out_folder OUT_FOLDER
                        Path to where to save embeddings
  --dropout DROPOUT     Dropout (default: 0.3)
  --input_dim INPUT_DIM
                        GNN input dim (default: 5)
  --hidden_dim HIDDEN_DIM
                        GNN hidden dim (default: 128)
  --output_dim OUTPUT_DIM
                        GNN output dim (default: 16)
  --num_rec NUM_REC     GNN rec (default: 5)
  --num_layers NUM_LAYERS
                        GNN layers (default: 2)
  --train               Train or test mode
```

## Generate logits
- First train the sampled architectures to generate training and testing logits for the training and testing idxs sampled before. Specifying 0 in the subset size selects the entire training set.
```bash
usage: train_full_partial.py [-h] [--subset_size SUBSET_SIZE]
                             [--dataset DATASET] [--root ROOT]
                             [--device DEVICE] [--learning_rate LEARNING_RATE]
                             [--weight_decay WEIGHT_DECAY]
                             [--out_folder OUT_FOLDER]
                             [--test_every TEST_EVERY] [--epochs EPOCHS]
                             [--seed SEED] [--batch_size BATCH_SIZE]
                             [--test_batch_size TEST_BATCH_SIZE]
                             [--json_folder JSON_FOLDER]
                             [--subset_folder SUBSET_FOLDER]
                             [--out_channels OUT_CHANNELS]
                             [--num_cells NUM_CELLS]
                             [--num_internal_cells NUM_INTERNAL_CELLS]

optional arguments:
  --subset_size SUBSET_SIZE, -s SUBSET_SIZE
                        Subset size (if 0: considers entire set)
  --dataset DATASET     Dataset name
  --root ROOT           Path to dataset
  --device DEVICE       cuda or cpu
  --learning_rate LEARNING_RATE
                        Learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --out_folder OUT_FOLDER
                        Path to save checkpoints
  --test_every TEST_EVERY
                        Eval loop frequency
  --epochs EPOCHS       Number of epochs
  --seed SEED           Seed for RNG
  --batch_size BATCH_SIZE
                        Training batch size
  --test_batch_size TEST_BATCH_SIZE
                        Testing batch_size
  --json_folder JSON_FOLDER
                        Path to json files
  --subset_folder SUBSET_FOLDER
                        Path to indices
  --out_channels OUT_CHANNELS
                        Channels in CNN
  --num_cells NUM_CELLS
                        Number of stacked cells externally
  --num_internal_cells NUM_INTERNAL_CELLS
                        Number of stacked cells internally
```
- Train approximator as
```bash
usage: train_encoder.py [-h] [--device DEVICE] --archemb_file ARCHEMB_FILE
                        --dataemb_file DATAEMB_FILE
                        [--logit_train_file LOGIT_TRAIN_FILE]
                        [--logit_test_file LOGIT_TEST_FILE]
                        [--logit_train_indices LOGIT_TRAIN_INDICES]
                        [--logit_test_indices LOGIT_TEST_INDICES]
                        [--load_checkpoint LOAD_CHECKPOINT]
                        [--save_checkpoint SAVE_CHECKPOINT]
                        [--experiment_name EXPERIMENT_NAME]
                        [--hidden_dim HIDDEN_DIM] [--arch_dim ARCH_DIM]
                        [--num_classes NUM_CLASSES]

optional arguments:
  --device DEVICE       cuda device id (default: 0)
  --archemb_file ARCHEMB_FILE
                        Path to architecture embeddings
  --dataemb_file DATAEMB_FILE
                        Path to data embeddings
  --logit_train_file LOGIT_TRAIN_FILE
                        Path to training logits
  --logit_test_file LOGIT_TEST_FILE
                        Path to testing logits
  --logit_train_indices LOGIT_TRAIN_INDICES
                        Path to architecture idxs for training set
  --logit_test_indices LOGIT_TEST_INDICES
                        Path to architecture idxs for testing set
  --load_checkpoint LOAD_CHECKPOINT
                        Load checkpoint to resume
  --save_checkpoint SAVE_CHECKPOINT
                        Path to save weights (default:
                        model_encoder_checkpoint.pt)
  --experiment_name EXPERIMENT_NAME
                        Name (default: model_encoder)
  --hidden_dim HIDDEN_DIM
                        Hidden dimension for FFN (default: 256)
  --arch_dim ARCH_DIM   Hidden dimension for embeddings (default: 16)
  --num_classes NUM_CLASSES
                        Number of classes in dataset (default: 10)
```
- Generate the approximated logits
```bash
usage: gen_logits.py [-h] [--device DEVICE] --archemb_file ARCHEMB_FILE
                     --dataemb_file DATAEMB_FILE --load_checkpoint
                     LOAD_CHECKPOINT [--experiment_name EXPERIMENT_NAME]

optional arguments:
  --device DEVICE
  --archemb_file ARCHEMB_FILE
                        Path to architecture embeddings
  --dataemb_file DATAEMB_FILE
                        Path to data embeddings
  --load_checkpoint LOAD_CHECKPOINT
                        Path to approx weights
  --experiment_name EXPERIMENT_NAME
                        Name
```

## Generate subsets
To generate transductive indices:
```bash
usage: tr_subset.py [-h] [--data_file DATA_FILE] [--targets_file TARGETS_FILE]
                    [--archemb_file ARCHEMB_FILE]
                    [--approx_checkpoint APPROX_CHECKPOINT]
                    [--dataset DATASET] [--json_folder JSON_FOLDER]
                    [--subset_size SUBSET_SIZE] [--num_iter NUM_ITER]
                    [--lambda_1 LAMBDA_1] [--lambda_2 LAMBDA_2]
                    [--learning_rate LEARNING_RATE]
                    [--output_folder OUTPUT_FOLDER] [--ma_dropout MA_DROPOUT]

optional arguments:
  --data_file DATA_FILE
                        Path to the data embeddings
  --targets_file TARGETS_FILE
                        Path to the targets
  --archemb_file ARCHEMB_FILE
                        Path to the architecture embeddings
  --approx_checkpoint APPROX_CHECKPOINT
                        Path to the approx weights
  --dataset DATASET     Name of the dataset
  --json_folder JSON_FOLDER
                        Path to the folder containing architecture json files
  --subset_size SUBSET_SIZE
                        Subset size (as integer)
  --num_iter NUM_ITER   Number of iterations
  --lambda_1 LAMBDA_1   Entropy weightage (default: 1)
  --lambda_2 LAMBDA_2   Regularizer weightage (default: 0.1)
  --learning_rate LEARNING_RATE
  --output_folder OUTPUT_FOLDER
                        Path to save the generated indices
  --ma_dropout MA_DROPOUT
```
- To generate inductive indices:
```bash
usage: ind_subset.py [-h] [--data_file DATA_FILE]
                     [--targets_file TARGETS_FILE] --y_onehot_file
                     Y_ONEHOT_FILE [--archemb_file ARCHEMB_FILE]
                     [--model_encoder_file MODEL_ENCODER_FILE]
                     [--subset_size SUBSET_SIZE] [--dataset DATASET]
                     [--json_folder JSON_FOLDER] [--num_iter NUM_ITER]
                     [--lambda_1 LAMBDA_1] [--lambda_2 LAMBDA_2]
                     [--learning_rate LEARNING_RATE]
                     [--output_folder OUTPUT_FOLDER]
                     [--num_classes NUM_CLASSES]

optional arguments:
  --data_file DATA_FILE
                        Path to the data embeddings
  --targets_file TARGETS_FILE
                        Path to the targets
  --y_onehot_file Y_ONEHOT_FILE
                        Path to one-hot targets
  --archemb_file ARCHEMB_FILE
                        Path to the architecture embeddings
  --model_encoder_file MODEL_ENCODER_FILE
                        Path to the approx predictions
  --subset_size SUBSET_SIZE
                        Subset size (as integer)
  --dataset DATASET     Name of the dataset
  --json_folder JSON_FOLDER
                        Path to the folder containing architecture json files
  --num_iter NUM_ITER   Number of iterations
  --lambda_1 LAMBDA_1   Entropy weightage (default: 1)
  --lambda_2 LAMBDA_2   Regularizer weightage (default: 0.1)
  --learning_rate LEARNING_RATE
  --output_folder OUTPUT_FOLDER
                        Path to save the generated indices
  --num_classes NUM_CLASSES
                        Number of classes in dataset (default: 10)
```
- Then train the model on the chosen indices by specifying `subset_folder` and `subset_size` in `train_full_partial.py`

## Citation
```bibtex
@inproceedings{
jain2023efficient,
title={Efficient Data Subset Selection to Generalize Training Across Models: Transductive and Inductive Networks},
author={Eeshaan Jain and Tushar Nandy and Gaurav Aggarwal and Ashish V. Tendulkar and Rishabh K Iyer and Abir De},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=q3fCWoC9l0}
}
```
