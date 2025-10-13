import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["spectrogram"] = [item['spectrogram'][0] for item in dataset_items]
    result_batch["x"] = pad_sequence([item["spectrogram"][0].transpose(0, 1) for item in dataset_items], batch_first=True)
    result_batch["text_encoded"] = pad_sequence([item["text_encoded"][0] for item in dataset_items], batch_first=True).int()
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["audio"] = [item["audio"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    result_batch["spectrogram_length"] = torch.tensor([item["spectrogram"][0].shape[1] for item in dataset_items], dtype=torch.int32)
    result_batch["text_encoded_length"] = torch.tensor([len(item["text_encoded"][0]) for item in dataset_items], dtype=torch.int32)

    return result_batch
