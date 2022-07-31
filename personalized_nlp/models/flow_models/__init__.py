from personalized_nlp.models.flow_models.flow_baseline import FlowBaseline
from personalized_nlp.models.flow_models.flow_onehot import FlowOneHot


FLOW_MODELS_DICT = {
    'flow_baseline': FlowBaseline,
    'flow_onehot': FlowOneHot
}