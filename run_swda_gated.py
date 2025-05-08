#!/usr/bin/env python3
import os

# Script to train the gated-fusion variant on the SWDA corpus
if __name__ == '__main__':
    corpus = 'swda'
    mode = 'train'  # or 'inference'
    batch_size = 60
    batch_size_val = 32  # keep evaluation batch size larger
    emb_batch = 0        # use single-batch embedding
    epochs = 5          # number of training epochs
    gpu = '0,1'          # GPUs to use; adjust as needed
    lr = 2e-5            # learning rate
    nlayer = 2          # number of GRU layers
    chunk_size = 196     # chunk size for conversation slicing
    dropout = 0.5        # dropout probability
    nfinetune = 1        # number of RoBERTa layers to finetune
    nclass = 43         # number of dialogue-act classes

    # Use the new gating fusion mechanism (must be implemented in models.py)
    speaker_info = 'gated'
    topic_info = 'none'

    # Create output directory
    results_dir = f'results_{corpus}_gated'
    os.makedirs(results_dir, exist_ok=True)

    # Assemble and run command
    command = (
        f"python -u engine.py"
        f" --corpus={corpus}"
        f" --mode={mode}"
        f" --gpu={gpu}"
        f" --batch_size={batch_size}"
        f" --batch_size_val={batch_size_val}"
        f" --epochs={epochs}"
        f" --lr={lr}"
        f" --nlayer={nlayer}"
        f" --chunk_size={chunk_size}"
        f" --dropout={dropout}"
        f" --nfinetune={nfinetune}"
        f" --speaker_info={speaker_info}"
        f" --topic_info={topic_info}"
        f" --nclass={nclass}"
        f" --emb_batch={emb_batch}"
    )

    print("Running command:\n", command)
    os.system(command)
