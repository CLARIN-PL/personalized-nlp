from personalized_nlp.models.baseline import Net
from personalized_nlp.models.onehot import NetOneHot
from personalized_nlp.models.human_bias import HumanBiasNet
from personalized_nlp.models.bias import AnnotatorBiasNet
from personalized_nlp.models.past_embeddings import PastEmbeddingsNet
from personalized_nlp.models.word_bias import WordBiasNet
from personalized_nlp.models.annotator import AnnotatorEmbeddingNet
from personalized_nlp.models.annotator_word import AnnotatorWordEmbeddingNet
from personalized_nlp.models.user_id import NetUserID
from personalized_nlp.models.hubi_med_finetune import HuBiMedium

models = {
    'baseline': Net,
    'onehot': NetOneHot,
    'peb': HumanBiasNet,
    'word_bias': WordBiasNet,
    'bias': AnnotatorBiasNet,
    'embedding': AnnotatorEmbeddingNet,
    'word_embedding': AnnotatorWordEmbeddingNet,
    'past_embeddings': PastEmbeddingsNet,
    'userid': NetUserID,
    'hubi_med': HuBiMedium
}
