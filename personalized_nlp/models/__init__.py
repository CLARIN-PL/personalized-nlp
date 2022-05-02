from personalized_nlp.models.baseline import Baseline
from personalized_nlp.models.onehot import NetOneHot
from personalized_nlp.models.hubi_formula import HuBiFormula
from personalized_nlp.models.hubi_simple import HuBiSimple
from personalized_nlp.models.hubi_medium import HuBiMedium
from personalized_nlp.models.transformer import TransformerUserId

models = {
    "baseline": Baseline,
    "onehot": NetOneHot,
    "peb": HuBiFormula,
    "bias": HuBiSimple,
    "embedding": HuBiMedium,
    "transformer_user_id": TransformerUserId,
}
