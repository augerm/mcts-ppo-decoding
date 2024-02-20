import unittest
from unittest.mock import MagicMock
from mcts import treeNode, MCTS

class MockGameState:
    def __init__(self, terminal=False, possible_actions=None, reward=0):
        self.terminal = terminal
        self.possible_actions = possible_actions or []
        self.reward = reward

    def isTerminal(self):
        return self.terminal

    def getPossibleActions(self):
        return self.possible_actions

    def takeAction(self, action):
        # Simply return a new state with decremented actions for simplicity
        return MockGameState(terminal=len(self.possible_actions) == 1, possible_actions=self.possible_actions[:-1], reward=self.reward)

    def getReward(self):
        return self.reward

    def getCurrentPlayer(self):
        # Mock player for simplicity, 1 for player 1's turn, -1 for player 2's turn
        return 1

class TestMCTS(unittest.TestCase):
    def test_treeNode_initialization(self):
        state = MockGameState()
        node = treeNode(state, None)
        self.assertEqual(node.state, state)
        self.assertFalse(node.isTerminal)
        self.assertFalse(node.isFullyExpanded)
        self.assertIsNone(node.parent)
        self.assertEqual(node.numVisits, 0)
        self.assertEqual(node.totalReward, 0)
        self.assertEqual(len(node.children), 0)

    def test_mcts_initialization(self):
        mcts_instance = MCTS(timeLimit=1000)
        self.assertEqual(mcts_instance.timeLimit, 1000)
        self.assertEqual(mcts_instance.limitType, 'time')
        with self.assertRaises(ValueError):
            MCTS(timeLimit=1000, iterationLimit=100)

    def test_selectNode_notFullyExpanded(self):
        state = MockGameState(possible_actions=['a', 'b'])
        mcts_instance = MCTS(iterationLimit=100)  # Add an iterationLimit
        node = treeNode(state, None)
        selected_node = mcts_instance.selectNode(node)
        self.assertIn(selected_node, node.children.values())

    def test_expand(self):
        state = MockGameState(possible_actions=['a', 'b'])
        mcts_instance = MCTS(iterationLimit=100)  # Add an iterationLimit
        node = treeNode(state, None)
        mcts_instance.expand(node)
        self.assertIn('a', node.children.keys() or 'b' in node.children.keys())
        self.assertFalse(node.isFullyExpanded)

    def test_backpropogate(self):
        state = MockGameState(possible_actions=['a'], reward=1)
        root = treeNode(state, None)
        child = treeNode(state.takeAction('a'), root)
        mcts_instance = MCTS(iterationLimit=100)  # Add an iterationLimit
        mcts_instance.backpropogate(child, 1)
        self.assertEqual(root.numVisits, 1)
        self.assertEqual(root.totalReward, 1)
        self.assertEqual(child.numVisits, 1)
        self.assertEqual(child.totalReward, 1)

if __name__ == '__main__':
    unittest.main()
