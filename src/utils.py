
# Function to create batches
def batchify(data, batch_size):
    """Split data into batches of size `batch_size`."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
