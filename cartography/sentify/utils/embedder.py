from pathlib import Path


class BackboneEmbedder:
    def __init__(self, embedding_store: Path = Path('./embeddings')) -> None:
        self._embedding_store = embedding_store
        self._embedding_store.mkdir(exist_ok=True, parents=True)
        self._index_path = self._embedding_store.joinpath('index.json')

    def __call__(self, input_ids, *args, **kwargs) -> None:
        pass
