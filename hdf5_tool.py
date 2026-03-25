import h5py
import numpy as np
from tqdm import tqdm

def rechunk_h5(
    input_path,
    output_path,
    batch_size,
    data_keys=(
        "cur_emb",
        "hist_white_emb",
        "hist_black_emb",
        "hist_white_len",
        "hist_black_len",
        "side_to_move",
        "from_sq",
        "to_sq",
        "label",
        "promotion",
    ),
):
    """
    Reads an H5 file, shuffles the indices, and writes a new chunked file.
    """
    with h5py.File(input_path, 'r') as f_in:
        # 1. Setup metadata
        num_samples = len(f_in[data_keys[0]])
        indices = np.arange(num_samples)
        np.random.shuffle(indices) # Global pre-shuffle
        
        print(f"Re-chunking {num_samples} samples with chunk_size={batch_size}...")

        with h5py.File(output_path, 'w') as f_out:
            # 2. Create datasets with explicit chunking
            datasets = {}
            for k in data_keys:
                shape = f_in[k].shape
                dtype = f_in[k].dtype
                # We set chunks=batch_size so one batch = exactly one disk read
                chunks = (batch_size,) + shape[1:] 
                datasets[k] = f_out.create_dataset(
                    k, shape=shape, dtype=dtype, chunks=chunks, compression="lzf"
                )

            # 3. Buffered Write (to avoid RAM explosion)
            # We process in 'write_buffer' blocks to keep sequential writes fast
            write_buffer = batch_size * 100 
            for i in tqdm(range(0, num_samples, write_buffer)):
                end = min(i + write_buffer, num_samples)
                batch_indices = indices[i:end]
                
                # Sort indices for the input read (HDF5 reads faster if indices are sorted)
                sort_idx = np.argsort(batch_indices)
                rev_sort_idx = np.argsort(sort_idx)
                
                for k in data_keys:
                    # Read shuffled data from source
                    data_chunk = f_in[k][np.sort(batch_indices)]
                    # Place it into the new file in the new order
                    datasets[k][i:end] = data_chunk[rev_sort_idx]

    print(f"Done! Saved to {output_path}")

# Run it
rechunk_h5("databases/moves/500k_mayfly/train.h5", "optimized_data_1024.h5", batch_size=1024)
