import torch
import unittest
from unittest.mock import MagicMock, patch
from transformer_state import TransformerState, TransformerAction

class TestTransformerState(unittest.TestCase):
    def setUp(self):
        # Mocking the model and other parameters for initialization
        self.model_mock = MagicMock()
        self.x_mock = torch.tensor([1, 2, 3])  # Example sequence tensor
        self.freqs_cis_mock = MagicMock()
        self.cache_mock = MagicMock()
        self.p_value = 0.9
        self.state = TransformerState(self.model_mock, self.x_mock, self.freqs_cis_mock, self.cache_mock, self.p_value)
    
    def tearDown(self):
        # Reset the state to None or reconstruct it if necessary
        self.state = None

        # Reset mocks if needed
        self.model_mock.reset_mock()

    def test_get_current_player(self):
        # Assert that getCurrentPlayer returns 1 when the current player is the maximizer
        self.assertEqual(self.state.getCurrentPlayer(), 1)

    def test_get_possible_actions(self):
        # Mock the model's forward pass to return probabilities for some tokens
        mock_probabilities = torch.tensor([0.025, 0.025, 0.05, 0.4, 0.5])
        self.model_mock.forward.return_value = [mock_probabilities]
        self.state.p = 0.6  # Set a lower p value for easier testing

        # Call getPossibleActions
        possible_actions = self.state.getPossibleActions()
        possible_actions.sort()
        
        # Assert that the returned actions are correct based on nucleus sampling
        expected_actions = [TransformerAction(token_id) for token_id in [3, 4]]  # Tokens with probabilities above threshold
        expected_actions.sort()

        self.assertListEqual(possible_actions, expected_actions)
    
    def test_take_action(self):
        # Initial sequence length
        initial_length = self.x_mock.shape[0]

        # Create a new action to take
        new_action = TransformerAction(4)  # Arbitrary token_id for the test

        # Take the action and get the new state
        new_state = self.state.takeAction(new_action)

        # Assert the new sequence is longer by 1 token
        self.assertEqual(new_state.x.shape[0], initial_length + 1)

        # Assert the last token in the new sequence is the token_id of the action taken
        self.assertEqual(new_state.x[-1].item(), new_action.token_id)

        # Assert that the new state is indeed a different instance
        self.assertNotEqual(id(self.state), id(new_state))

    def test_is_terminal(self):
        # Case 1: Mock model to return probabilities indicating a non-terminal state
        mock_probabilities_non_terminal = torch.tensor([0.05, 0.15, 0.2, 0.4, 0.2])
        self.model_mock.forward.return_value = [torch.log_softmax(mock_probabilities_non_terminal, dim=-1)]
        self.state.min_prob_threshold = 0.1
        self.assertFalse(self.state.isTerminal(), "State should not be terminal when max probability is above the threshold.")

        # Case 2: Mock model to return probabilities indicating a terminal state
        mock_probabilities_terminal = torch.tensor([0.01, 0.01, 0.01, 0.02, 0.95])  # High probability concentrated in one token
        self.model_mock.forward.return_value = [torch.log_softmax(mock_probabilities_terminal, dim=-1)]
        self.state.min_prob_threshold = 0.95  # Adjust threshold for test
        self.assertTrue(self.state.isTerminal(), "State should be terminal when max probability is below the threshold.")
    
    def test_get_reward(self):
        # High confidence scenario
        self.model_mock.forward.return_value = torch.tensor([[10.0, 1.0, -1.0, -2.0]]) # Logits that softmax to high confidence for the top token
        high_confidence_reward = self.state.getReward()
        self.assertEqual(high_confidence_reward, 1.0, "High confidence should yield a high reward.")

        # Medium confidence scenario
        self.model_mock.forward.return_value = torch.tensor([[2.0, 1.5, 0.8]])  # Logits that softmax to medium confidence for the top token
        medium_confidence_reward = self.state.getReward()
        self.assertEqual(medium_confidence_reward, 0.5, "Medium confidence should yield a medium reward.")

        # Low confidence scenario
        self.model_mock.forward.return_value = torch.tensor([[2.0, 1.8, 1.5, 1.2]])  # Logits that softmax to low confidence for all tokens
        low_confidence_reward = self.state.getReward()
        self.assertEqual(low_confidence_reward, 0.0, "Low confidence should yield a low reward.")

if __name__ == '__main__':
    unittest.main()
