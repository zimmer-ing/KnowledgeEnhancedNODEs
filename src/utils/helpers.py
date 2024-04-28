import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def dl_is_shuffle(dataloader):
    """ Check if a DataLoader is shuffled or not."""
    # Check the type of sampler being used by the DataLoader
    if isinstance(dataloader.sampler, RandomSampler):
        return True
    elif isinstance(dataloader.sampler, SequentialSampler):
        return False
    # For BatchSampler, check if the underlying sampler is a RandomSampler
    elif hasattr(dataloader, 'batch_sampler') and isinstance(dataloader.batch_sampler.sampler, RandomSampler):
        return True
    else:
        # If none of the above, it's unclear or not shuffled
        return False


def change_to_sequential_and_return_new(dataloader):
    """ Change a DataLoader to use SequentialSampler and return a new DataLoader."""
    # Create a new DataLoader with SequentialSampler
    new_dataloader = DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # Set shuffle to False to use SequentialSampler
        sampler=SequentialSampler(dataloader.dataset),  # Explicitly set to SequentialSampler
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers
    )

    # Check if the new DataLoader is in sequential mode using the is_shuffle function
    if not dl_is_shuffle(new_dataloader):
        print("Successfully created a new DataLoader in sequential mode.")
    else:
        print("Failed to create a new DataLoader in sequential mode.")

    return new_dataloader

def ensure_sequential_dataloader(dataloader):
    """
    Ensure that a DataLoader is in sequential mode by creating a new DataLoader with SequentialSampler.

    Parameters:
    - dataloader: The DataLoader to check and possibly change.

    Returns:
    - A new DataLoader with SequentialSampler if the original DataLoader was shuffled,
      otherwise the original DataLoader is returned.
    """
    # Check if the DataLoader is already in sequential mode
    if not dl_is_shuffle(dataloader):
        print("DataLoader is already in sequential mode.")
        return dataloader

    # Create a new DataLoader with SequentialSampler
    new_dataloader = change_to_sequential_and_return_new(dataloader)

    return new_dataloader


import torch


def concatenate_batches(predictions):
    """
    Concatenate a list of batched tensors into a single tensor with shape [time, features].

    Parameters:
    - predictions: A list of tensors with shape [batch, time, features].

    Returns:
    - A tensor with shape [total_time, features], where total_time is the sum of all time dimensions
      across the batches.
    """
    # Step 1: Concatenate all tensors in the list along the batch dimension
    concatenated = torch.cat(predictions, dim=0)  # Results in [total_batches * batch, time, features]

    # Calculate the new shape
    total_batches_times_batch, time, features = concatenated.shape

    # Step 2: Reshape to [total_time, features], merging the first two dimensions
    result = concatenated.view(-1, features)  # Merges batch and time dimensions

    return result




if __name__ == "__main__":

    #test the functions above
    # Example usage
    dataset = torch.rand(100, 2)  # Just a dummy dataset
    dataloader_shuffled = DataLoader(dataset, batch_size=10, shuffle=True)
    dataloader_sequential = DataLoader(dataset, batch_size=10, shuffle=False)

    print("Shuffled DataLoader:", dl_is_shuffle(dataloader_shuffled))
    print("Sequential DataLoader:", dl_is_shuffle(dataloader_sequential))

    # Example usage
    dataset = torch.rand(100, 2)  # Dummy dataset
    dataloader_shuffled = DataLoader(dataset, batch_size=10, shuffle=True)

    # Change DataLoader to sequential and check
    print("DataLoader is shuffle before :", dl_is_shuffle(dataloader_shuffled))
    dataloader_sequential = ensure_sequential_dataloader(dataloader_shuffled)
    print("DataLoader is shuffle after :", dl_is_shuffle(dataloader_sequential))

    ##### concat_batches function
    # Example usage
    # Single batch case, batch dimension size = 1
    batch1 = torch.randn(1, 5, 3)  # For example, 1 sequence of length 5 with 3 features
    # Multiple batch case
    batch2 = torch.randn(2, 5, 3)  # For example, 2 sequences of length 5 with 3 features each

    # Concatenate batches
    result_single = concatenate_batches([batch1])  # Handling single list element
    result_multiple = concatenate_batches([batch1, batch2])  # Handling multiple batches

    print("Single batch result shape:", result_single.shape)
    print("Multiple batches result shape:", result_multiple.shape)