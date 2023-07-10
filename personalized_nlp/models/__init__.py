from personalized_nlp.models.baseline import Baseline
from personalized_nlp.models.onehot import NetOneHot
from personalized_nlp.models.hubi_formula import HuBiFormula
from personalized_nlp.models.hubi_simple import HuBiSimple
from personalized_nlp.models.hubi_medium import HuBiMedium
from personalized_nlp.models.transformer import TransformerUserId
from personalized_nlp.models.transformer_multihead import TransformerMultiUserHead
from personalized_nlp.models.transformer_multihead_single_text import (
    TransformerMultiUserSingleTextHead,
)

models = {
    "baseline": Baseline,
    "onehot": NetOneHot,
    "peb": HuBiFormula,
    "bias": HuBiSimple,
    "embedding": HuBiMedium,
    "transformer_user_id": TransformerUserId,
    "transformer_multihead": TransformerMultiUserHead,
    "transformer_multihead_single_text": TransformerMultiUserSingleTextHead,
}
