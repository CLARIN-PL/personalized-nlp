from personalized_nlp.models.flow_models.flow_baseline import FlowBaseline
from personalized_nlp.models.flow_models.flow_onehot import FlowOneHot
from personalized_nlp.models.flow_models.flow_peb import FlowPEB
from personalized_nlp.models.flow_models.flow_bias import FlowBias


FLOW_MODELS_DICT = {
    'flow_baseline': FlowBaseline,
    'flow_onehot': FlowOneHot,
    'flow_peb': FlowPEB,
    'flow_bias': FlowBias
}
