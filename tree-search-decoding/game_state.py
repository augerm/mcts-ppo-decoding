import torch

class GameState:
    def __init__(self, model, tokenizer, sequence, temperature=0.7, top_k=5):
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
        self.player_to_move = 1  # Assuming a single-player setup

    def get_legal_actions(self):
        """
        Get top k tokens as legal actions based on the model's predictions.
        """
        # Convert the current sequence to a tensor and perform a forward pass through the model
        with torch.no_grad():
            inputs = torch.tensor([self.sequence], dtype=torch.long, device=self.model.device)
            logits = self.model.forward(inputs)[-1, :, :]  # Get logits for the last token in the sequence
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            # Convert logits to probabilities
            probs = torch.softmax(scaled_logits, dim=-1)
            # Select the top k tokens based on their probabilities
            top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
            return top_indices[0].cpu().tolist()

    def do_action(self, action):
        """
        Perform an action by appending the selected token to the current sequence.
        """
        self.sequence.append(action)

    def clone(self):
        """
        Return a deep copy of the current game state.
        """
        return GameState(self.model, self.tokenizer, self.sequence.copy(), self.temperature, self.top_k)

    def get_result(self, player_to_move):
        """
        Compute and return the game result. Placeholder for a more sophisticated result evaluation.
        """
        # Simplified result computation. In practice, you'll need a meaningful way to evaluate the outcome.
        return 1 if True else 0
