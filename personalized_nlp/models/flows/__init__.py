from personalized_nlp.models.flows.cnice import cNICE
from personalized_nlp.models.flows.creal_nvp import cRealNVP
from personalized_nlp.models.flows.cmaf import cMAF


FLOWS_DICT = {
    'nice': cNICE,
    'real_nvp': cRealNVP,
    'maf': cMAF
}
