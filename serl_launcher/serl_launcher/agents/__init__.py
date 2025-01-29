from .continuous.bc import BCAgent
from .continuous.sac import SACAgent
from .continuous.bc_noimg import BCAgentNoImg

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
}
