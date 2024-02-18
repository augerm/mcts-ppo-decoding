import random
import torch

from tree_search_decoding.game_state import GameState

class PolicyModel:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def __init__(self):
    # Initialize your policy model here (e.g., neural network weights)
    pass

  def predict(self, game_state: GameState):
    """
    Predicts the best action to take in a given game state.
    This is a stub, replace with your own logic or model.
    """
    # Implement your policy model's prediction logic here
    print(Warning('Policy model should be returning probability distributions, but not yet implemented!'))
    return game_state.get_legal_actions()
