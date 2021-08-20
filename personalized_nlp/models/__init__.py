
from personalized_nlp.models.baseline import Net
from personalized_nlp.models.onehot import NetOneHot
from personalized_nlp.models.human_bias import HumanBiasNet
from personalized_nlp.models.bias import AnnotatorBiasNet
from personalized_nlp.models.annotator import AnnotatorEmbeddingNet
from personalized_nlp.models.annotator_word import AnnotatorWordEmbeddingNet

models = {
    'baseline': Net,
    'onehot': NetOneHot, 
    'peb': HumanBiasNet, 
    'bias': AnnotatorBiasNet, 
    'embedding': AnnotatorEmbeddingNet, 
    'word_embedding': AnnotatorWordEmbeddingNet
}