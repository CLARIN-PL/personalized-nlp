# TODO: UGLY! Refactor but I doubt we will have time for that
import pickle
from pathlib import Path

import torch
from pytorch_lightning import Callback
from tqdm import tqdm

from personalized_nlp.models.transformer import TransformerUserId
from personalized_active_learning.learning.training import train_test
from settings import TRANSFORMER_MODEL_STRINGS


def fine_tune_embeddings(
    original_dataset,  # type is BaseDataset, cannot be imported due to circular import
    batch_size: int = 20,
    epochs=3,
    lr=2e-5,
):
    use_cuda = original_dataset.embeddings_creator.use_cuda
    embeddings_type = original_dataset.embeddings_creator.embeddings_type
    split_mode_str = original_dataset.split_mode.value
    data_dir = original_dataset.embeddings_creator.directory
    test_fold_index = original_dataset.test_fold_index

    embeddings_path = data_dir / f"{embeddings_type}_{test_fold_index}_{split_mode_str}.p"
    # IMPORTANT we need to change path in original creator
    original_dataset.embeddings_creator.path = embeddings_path
    # Embedding already created
    if embeddings_path.exists():
        return
    # TODO: UGLY! Refactor but I doubt we will have time for that
    dataset_cls = type(original_dataset)
    init_kwargs = original_dataset.init_kwargs
    init_kwargs["major_voting"] = True
    init_kwargs["batch_size"] = batch_size
    init_kwargs["use_finetuned_embeddings"] = False

    dataset = dataset_cls(**init_kwargs)
    model = TransformerUserId(
        text_embedding_dim=original_dataset.embeddings_creator.text_embedding_dim,
        output_dim=sum(original_dataset.classes_dimensions),
        huggingface_model_name=TRANSFORMER_MODEL_STRINGS[embeddings_type],
        max_length=128,
        append_annotator_ids=False,
        annotator_num=original_dataset.annotators_number,
        use_cuda=use_cuda,
    )

    train_test(
        dataset,
        model=model,
        epochs=epochs,
        lr=lr,
        use_cuda=use_cuda,
        custom_callbacks=[
            SaveEmbeddingCallback(
                dataset=dataset,
                save_path=embeddings_path,
            )
        ],
    )


class SaveEmbeddingCallback(Callback):
    def __init__(
        self,
        dataset,  # type is BaseDataset, cannot be imported due to circular import
        save_path: Path,
    ):
        self.dataset = dataset
        self.save_path = save_path

    def on_test_end(self, trainer, pl_module) -> None:
        # TODO: Access to protected attributes
        model = pl_module.model._model
        tokenizer = pl_module.model._tokenizer
        texts = self.dataset.data["text"].tolist()

        embeddings = _get_embeddings(texts, tokenizer, model, use_cuda=True)
        embeddings = embeddings.cpu().numpy()

        text_idx_to_emb = {}
        for i in range(embeddings.shape[0]):
            text_idx_to_emb[i] = embeddings[i]

        embeddings_path = self.save_path
        if embeddings_path:
            pickle.dump(text_idx_to_emb, open(embeddings_path, "wb"))


def _get_embeddings(texts, tokenizer, model, max_seq_len=256, use_cuda=False):
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"

    all_embeddings = []
    for batched_texts in tqdm(batch(texts, 200), total=len(texts) / 200):
        with torch.no_grad():
            batch_encoding = tokenizer.batch_encode_plus(
                batched_texts,
                padding="longest",
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(device)

            emb = model(**batch_encoding)

        mask = batch_encoding["attention_mask"] > 0

        for i in range(emb[0].size()[0]):
            all_embeddings.append(emb[0][i, mask[i] > 0, :].mean(axis=0)[None, :])

    return torch.cat(all_embeddings, dim=0).to("cpu")
