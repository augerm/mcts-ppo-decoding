import math
import random

class MCTSNode:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.game_state.get_legal_actions()
        self.player_to_move = self.game_state.player_to_move

    def ucb1_select_child(self):
        """Select a child node using UCB1 policy."""
        C = 1.41  # Exploration parameter
        chosen_child = max(self.children, key=lambda x: x.wins / x.visits + C * math.sqrt(2 * math.log(self.visits) / x.visits))
        return chosen_child

    def add_child(self, action, game_state):
        """Add a new child node for the given action."""
        child_node = MCTSNode(game_state=game_state, parent=self)
        self.untried_actions.remove(action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        """Update this node's data from the child node."""
        self.visits += 1
        self.wins += result

def mcts(root_state, iterations):
    root_node = MCTSNode(game_state=root_state)

    for _ in range(iterations):
        node = root_node
        state = root_state.clone()

        # Selection
        while node.untried_actions == [] and node.children != []:  # node is fully expanded and non-terminal
            node = node.ucb1_select_child()
            state.do_action(node.game_state)

        # Expansion
        if node.untried_actions != []:
            action = random.choice(node.untried_actions)
            state.do_action(action)
            node = node.add_child(action, state)  # add child and descend tree

        # Simulation
        while state.get_legal_actions() != []:  # while state is non-terminal
            state.do_action(random.choice(state.get_legal_actions()))

        # Backpropagation
        while node is not None:
            node.update(state.get_result(node.player_to_move))  # backpropagate from the expanded node and work back to the root node
            node = node.parent

    return max(root_node.children, key=lambda c: c.visits).game_state  # return the move that was most visited