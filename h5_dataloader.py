import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler


class ShardedH5Dataset(Dataset):
    """
    A reusable HDF5 dataset that opens the file per-worker to avoid 
    locking issues and handles multiple data keys.
    """
    def __init__(self, file_path, data_keys=('data', 'labels')):
        self.file_path = file_path
        self.data_keys = data_keys
        self.archive = None
        
        # Determine total length once in the main process
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f[self.data_keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Important: Open the file inside the worker process
        if self.archive is None:
            self.archive = h5py.File(self.file_path, 'r', swmr=True)
        
        # Handle single index or list of indices (BatchSampler passes a list)
        if isinstance(index, (list, np.ndarray)):
            # Efficient slicing for contiguous indices
            start, end = index[0], index[-1] + 1
            assert False, ("Bad index", index)
            return {k: torch.from_numpy(np.array(self.archive[k][start:end])) for k in self.data_keys}
        
        # assert False, ("Bad index", index)
        return {k: torch.from_numpy(np.array(self.archive[k][index])) for k in self.data_keys}

class StaticShardedBatchSampler(Sampler):
    """
    Unweaves the dataset into 'num_workers' independent lanes.
    Each worker handles a unique, contiguous section of the file.
    """
    def __init__(self, dataset_len, num_workers, batch_size):
        self.num_samples = dataset_len
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Calculate samples per worker (dropping remainders for clean shards)
        self.samples_per_worker = self.num_samples // num_workers
        self.batches_per_worker = self.samples_per_worker // batch_size

    def __iter__(self):
        # We iterate through the 'batch slots' first
        for b_idx in range(self.batches_per_worker):
            # Then we yield one batch for each worker in round-robin order
            for w_idx in range(self.num_workers):
                offset = w_idx * self.samples_per_worker
                start = offset + (b_idx * self.batch_size)
                # We yield a list of indices which the Dataset will receive
                yield list(range(start, start + self.batch_size))

    def __len__(self):
        return self.batches_per_worker * self.num_workers


class SimpleH5Dataset(Dataset):
    """
    A simple, robust H5 dataset. 
    Handles worker-safe file opening and basic slicing.
    """
    def __init__(
        self,
        file_path,
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
        self.file_path = file_path
        self.data_keys = data_keys
        self.archive = None
        
        # Get length once in main process
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f[self.data_keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Open file locally in the worker process if not already open
        if self.archive is None:
            # swmr=True is safer for concurrent reads
            self.archive = h5py.File(self.file_path, 'r', swmr=True)
        
        # PyTorch might pass a single int or a list/slice depending on setup
        if isinstance(index, (list, np.ndarray, slice)):
            return {
                k: torch.from_numpy(np.array(self.archive[k][index])) 
                for k in self.data_keys
            }
        
        # Single item access
        return {
            k: torch.from_numpy(np.array(self.archive[k][index])) 
            for k in self.data_keys
        }

def get_simple_loader(file_path, batch_size=128, num_workers=4, shuffle=True):
    dataset = SimpleH5Dataset(file_path)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,      # Keeps GPU transfers fast
        drop_last=True,       # Prevents uneven small batches at the end
        prefetch_factor=2     # Each worker keeps 2 batches ready
    )

# --- Usage Example ---
if __name__ == "__main__":
    #FILE_PATH = "optimized_data_128.h5"
    FILE_PATH = "databases/moves/500k_mayfly/train.h5"
    BATCH_SIZE = 2048
    NUM_WORKERS = 2
    
    if False:
        # 1. Initialize Dataset
        dataset = ShardedH5Dataset(
            FILE_PATH,
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
        )
        
        # 2. Initialize the Sampler
        sampler = StaticShardedBatchSampler(
            dataset_len=len(dataset), 
            num_workers=NUM_WORKERS, 
            batch_size=BATCH_SIZE
        )
        
        # 3. Create DataLoader
        # Note: batch_size=1 is used because the Sampler already returns full batches
        loader = DataLoader(
            dataset, 
            batch_sampler=sampler, 
            num_workers=NUM_WORKERS,
            pin_memory=True  # Essential for GPU transfer speed
        )
        print(f"Lanes created: {NUM_WORKERS} | Samples per lane: {sampler.samples_per_worker}")
    else:
        loader = get_simple_loader(FILE_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)



    import time
    def benchmark_loader(loader, simulated_gpu_time=0.1, num_batches=50):
        """
        Simulates a training loop to measure if the DataLoader is the bottleneck.
        
        Args:
            loader: The DataLoader instance
            simulated_gpu_time: Seconds to sleep per batch (simulates forward/backprop)
            num_batches: How many batches to test
        """
        print(f"\n--- Starting Benchmark (Workers: {loader.num_workers}) ---")
        
        start_time = time.perf_counter()
        batch_times = []
        
        # We use a manual iterator to track the very first "cold start" load
        it = iter(loader)
        
        for i in range(num_batches):
            fetch_start = time.perf_counter()
            
            try:
                # This is where the main process blocks if workers aren't ready
                batch = next(it)
            except StopIteration:
                break
                
            fetch_end = time.perf_counter()
            wait_time = fetch_end - fetch_start
            batch_times.append(wait_time)
            
            # Simulate GPU Work (e.g., training on a batch)
            # In a perfect pipeline, wait_time should be nearly 0 after the first batch
            time.sleep(simulated_gpu_time)
            
            if i % 10 == 0 and i > 0:
                avg_wait = sum(batch_times[-10:]) / 10
                print(f"Batch {i:03d} | Avg Wait for Loader: {avg_wait:.4f}s")

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        total_samples = num_batches * loader.batch_sampler.batch_size
        
        print("-" * 40)
        print(f"Total Time: {total_duration:.2f}s")
        print(f"Throughput: {total_samples / total_duration:.2f} samples/sec")
        print(f"Average Wait per Batch: {sum(batch_times)/len(batch_times):.4f}s")
        
        if sum(batch_times)/len(batch_times) > (simulated_gpu_time * 0.1):
            print("RESULT: ⚠️  GPU is starving. Increase num_workers or check disk I/O.")
        else:
            print("RESULT: ✅  Pipeline is saturated. GPU is the bottleneck.")

    # --- Execution ---
    # Assuming 'loader' is defined from the previous code block:
    benchmark_loader(loader, simulated_gpu_time=0.05) # 50ms per "training step"