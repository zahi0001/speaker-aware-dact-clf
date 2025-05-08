#!/usr/bin/env python3
import os

# Script to train the gated-fusion variant on the MRDA corpus
if __name__ == '__main__':
    corpus = 'mrda'
    mode = 'train'            # train or inference
    batch_size = 60            # training batch size
    batch_size_val = 2        # evaluation batch size
    emb_batch = 256           # batch size for embedding layer
    epochs = 30              # number of epochs
    gpu = '0,1'               # GPUs to use
    lr = 1e-4                 # learning rate
    nlayer = 1                # GRU layers
    chunk_size = 350          # conversation chunk size
    dropout = 0.5             # dropout rate
    nfinetune = 1             # BERT layers to finetune
    nclass = 5                # number of dialog-act classes

    # Use the gated fusion mechanism
    speaker_info = 'gated'
    topic_info = 'none'

    # Prepare results directory
    results_dir = f'results_{corpus}_gated'
    os.makedirs(results_dir, exist_ok=True)

    # Build and run the command
    command = (
        f"python -u engine.py"
        f" --corpus={corpus}"
        f" --mode={mode}"
        f" --gpu={gpu}"
        f" --batch_size={batch_size}"
        f" --batch_size_val={batch_size_val}"
        f" --emb_batch={emb_batch}"
        f" --epochs={epochs}"
        f" --lr={lr}"
        f" --nlayer={nlayer}"
        f" --chunk_size={chunk_size}"
        f" --dropout={dropout}"
        f" --nfinetune={nfinetune}"
        f" --speaker_info={speaker_info}"
        f" --topic_info={topic_info}"
        f" --nclass={nclass}"
    )

    print("Running command:\n", command)
    os.system(command)
