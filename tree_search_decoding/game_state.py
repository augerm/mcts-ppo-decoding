import torch

from tree_search_decoding.policy_model import PolicyModel
from tree_search_decoding.value_model import ValueModel

class GameState:
    def __init__(self, model, policy_model: PolicyModel, value_model: ValueModel, tokenizer, sequence, last_token_prelogits, temperature=0.7, top_k=5, minimal_next_token_probability = 0.1):
        """
        Initialize the game state.
        :param model: The transformer model used for generating text.
        :param tokenizer: The tokenizer for encoding/decoding strings to/from token ids.
        :param sequence: The current sequence of token ids.
        :param temperature: Temperature for sampling from the model's output.
        :param top_k: Number of top tokens to consider as legal actions.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sequence = sequence
        self.temperature = temperature
        self.top_k = top_k
        self.policy_model = policy_model
        self.value_model = value_model
        self.last_token_prelogits = last_token_prelogits
        self.legal_actions = torch.softmax(last_token_prelogits / temperature, dim=-1)
        self.minimal_next_token_probability = minimal_next_token_probability

    def get_legal_actions(self):
        # Select the top k actions based on the policy model's probabilities
        top_probs, top_indices = torch.topk(self.legal_actions, self.top_k, dim=-1)
        if top_probs[0] > self.minimal_next_token_probability:
            return top_indices[0].cpu().tolist()
        else:
            return []

    def do_action(self, action):
        """
        Perform an action by appending the selected token to the current sequence.
        """
        self.sequence.append(action)

    def clone(self):
        """
        Return a deep copy of the current game state.
        """
        return GameState(self.model, self.policy_model, self.value_model, self.tokenizer, self.sequence.copy(), self.last_token_prelogits, self.temperature, self.top_k)

    def get_result(self):
        # Example of using a value model to compute the result
        # Assuming a separate value model that estimates the state value
        state_value = self.value_model.evaluate(self)
        return state_value.item()
