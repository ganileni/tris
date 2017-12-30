from scripts.agent_settings import menace_description, qlearning_description, random_description, \
    deepq_learning_description
from tris.agents import play_vs_human

available_agents = [menace_description,
                    qlearning_description,
                    deepq_learning_description,
                    random_description]

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
        play_vs_human(agent_instance, agent.name)
        answer = input('\nPlay again? (y/n) ').lower()
        if answer == 'n':
            break
