from personalized_nlp.models.baseline import Baseline
from personalized_nlp.models.onehot import NetOneHot
from personalized_nlp.models.hubi_formula import HuBiFormula
from personalized_nlp.models.hubi_simple import HuBiSimple
from personalized_nlp.models.hubi_medium import HuBiMedium
from personalized_nlp.models.transformer import TransformerUserId
from personalized_nlp.models.conformity import ConformityModel
from personalized_nlp.models.annotator_mean_text_embeddings import (
    AnnotatorMeanTextEmbeddingModel,
)
from personalized_nlp.models.attention import AttentionModel
from personalized_nlp.models.gru import GruModel

models = {
    "baseline": Baseline,
    "onehot": NetOneHot,
    "peb": HuBiFormula,
    "bias": HuBiSimple,
    "embedding": HuBiMedium,
    "transformer_user_id": TransformerUserId,
    "conformity": ConformityModel,
    "annotator_mean_text_embeddings": AnnotatorMeanTextEmbeddingModel,
    "attention": AttentionModel,
    "gru": GruModel,
}
