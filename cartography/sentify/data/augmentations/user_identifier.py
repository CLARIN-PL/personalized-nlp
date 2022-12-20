import abc
from abc import ABC

import torch
from transformers import PreTrainedTokenizerFast

from sentify.datasets.dataset import SampleType


class BaseUserIdentifierAugmentation(ABC):
    def __call__(self, item: SampleType) -> SampleType:
        username, text = item['username'], item['text']
        username = self._transform_username(username)
        text = f'{username} {text}'
        return {
            **item,
            'text': text,
        }

    @abc.abstractmethod
    def _transform_username(self, username: str) -> str:
        pass


class RandomUserIdentifierAugmentation(BaseUserIdentifierAugmentation):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        length: int = 10,
    ):
        self._tokenizer = tokenizer
        self._length = length
        self._vocab_start = len(self._tokenizer.additional_special_tokens_ids) + 1

    def _transform_username(self, username: str) -> str:
        username_hash = hash(username)
        ids = torch.randint(
            low=self._vocab_start,
            high=len(self._tokenizer.vocab),
            size=(self._length - 1,),
            generator=torch.Generator().manual_seed(username_hash),
        )
        return self._tokenizer.decode(ids, skip_special_tokens=True)


AUGMENTATIONS = {
    'random': RandomUserIdentifierAugmentation,
}
