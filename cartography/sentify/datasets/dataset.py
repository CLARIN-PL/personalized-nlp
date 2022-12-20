from typing import Any, Callable, Optional, TypedDict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

AugmentationType = Callable[[dict[str, Any]], dict[str, Any]]


class SampleType(TypedDict):
    index: int
    username: str
    text: str
    label: int
    split: str


class SentimentDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleType],
        tokenizer: PreTrainedTokenizerFast,
        split: str,
        augmentations: Optional[AugmentationType] = None,
        max_length: int = int(1e10),
    ):
        self._samples = samples
        self._tokenizer = tokenizer
        self._augmentations = augmentations or (lambda x: x)
        self._max_length = max_length
        self._split = split

    def __getitem__(self, idx: int):
        item = self._samples[idx]
        item['split'] = self._split
        item = self._augmentations(item)

        tokenized_text = self._tokenizer(
            text=item['text'],
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt',
            padding=False,  # no padding, doing later
            return_attention_mask=True,
        )
        str_val_keys = []
        for key in item:
            if type(item[key]) == str:
                str_val_keys.append(key)

        for key in str_val_keys:
            item.pop(key)

        return {
            **item,
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0),
        }

    def __len__(self):
        return len(self._samples)
