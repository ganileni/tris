from scripts.constants import AgentDescription
from scripts.train_agents import train_MENACE
from tris.agents import MENACEAgent

menace_description = AgentDescription(
        name = 'MENACE',
        filename='menace',
        klass = MENACEAgent,
        train_procedure = train_MENACE
)