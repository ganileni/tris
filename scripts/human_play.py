from scripts.agent_settings import menace_description
from tris.agents import HumanAgent
from tris.rules import Match

available_agents = [menace_description]

if __name__ == '__main__':
    while True:
        print('Welcome to Tris.\nPlease select the agent you would like to play against.\n')
        for n, agent in enumerate(available_agents):
            print('    ' + str(n) + '.  ', agent.name, 'agent')
        while True:
            print('\nenter a number: ', end='')
            choice = input()
            # basic input checking
            try:
                choice = int(choice)
            except ValueError:
                print("\n'" + choice + "' is not a number!\n,")
                continue
            try:
                agent = available_agents[choice]
            except IndexError:
                print('\nThere is no agent corresponding to ' + str(choice) + '!\n')
                continue
            try:
                agent_instance = agent.klass.load_agent(agent.path)
                break
            except FileNotFoundError:
                print("\nIt appears that " + agent.name + " agent has never been trained.\n")
                answer = input('Do you want to train it now? (y/n) ').lower()
                if answer == 'y':
                    # trains agent and saves to disk
                    agent.train(agent)
                    # instantiate from trained agent
                    agent_instance = agent.klass.load_agent(agent.path)
                    break
                else:
                    continue
        human = HumanAgent()
        match = Match(human, agent_instance)
        # set agent to maximum greed
        agent_instance.temperature = 0
        print('\nMatch started. ', end='')
        if not match.who_plays:
            print(agent.name + 'agent plays first.')
        else:
            print('You play first.')
        result = match.play()
        answer = input('\nPlay again? (y/n) ').lower()
        if answer == 'n':
            break
