import torch

class TransformerState():
    def __init__(self, model, x, freqs_cis, cache=None, p=0.9, min_prob_threshold=0.01):
        """
        Initialize the state with the necessary components for making predictions,
        including the nucleus sampling threshold p.
        """
        self.model = model
        self.x = x  # The current sequence of tokens (as a tensor)
        self.freqs_cis = freqs_cis
        self.cache = cache
        self.p = p
        self.min_prob_threshold = min_prob_threshold

    def getCurrentPlayer(self):
        """
        Returns the current player's identity.
        - Returns 1 for the maximizer (e.g., the entity trying to increase the score or probability).
        - Returns -1 for the minimizer (e.g., the entity trying to decrease the score or probability).
        """
        # We always want to maximize the next token's probability.
        return 1

    def getPossibleActions(self):
        """
        Uses the model's forward pass to predict the next set of possible actions,
        applying nucleus sampling to select a subset based on the cumulative probability threshold.
        """
        logits = self.model.forward(self.x, self.freqs_cis, self.cache)
        probabilities = torch.softmax(logits[-1], dim=-1)  # Focus on the last set of logits for the current step

        # Sort the probabilities in descending order and compute the cumulative probabilities
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)

        # Find the index where the cumulative probability exceeds the threshold p
        cutoff_index = (cumulative_probabilities > self.p).nonzero().min().item()

        # Select actions (token IDs) whose cumulative probability is within the threshold
        possible_actions = sorted_indices[:cutoff_index].tolist()
        
        # Convert token IDs to TransformerAction instances
        transformer_actions = [TransformerAction(token_id) for token_id in possible_actions]

        return transformer_actions

    def takeAction(self, action):
        """
        Updates the current sequence by appending the selected action (token).
        Returns a new TransformerState reflecting the updated sequence.
        
        Args:
            action (TransformerAction): The action to append.
        
        Returns:
            TransformerState: A new state with the updated sequence.
        """
        # Extract the token index from the TransformerAction instance.
        token_index = action.token_id

        # Append the action (token index) to the current sequence.
        # Assuming `self.x` is a tensor with shape [sequence_length] and `token_index` is a scalar.
        new_x = torch.cat((self.x, torch.tensor([token_index], dtype=self.x.dtype)), dim=0)

        # Create a new TransformerState with the updated sequence.
        # Note: You might need to update `freqs_cis` and `cache` as appropriate for your model.
        new_state = TransformerState(self.model, new_x, self.freqs_cis, self.cache, self.p)
        
        return new_state

    def isTerminal(self):
        logits = self.model.forward(self.x, self.freqs_cis, self.cache)
        probabilities = torch.softmax(logits[-1], dim=-1)
        max_prob = torch.max(probabilities)
        # If the maximum probability among the next token predictions is below the threshold,
        # consider the sequence terminal.
        return max_prob.item() < self.min_prob_threshold

    def getReward(self):
        # Perform the model's forward pass to get the logits for the next possible tokens.
        logits = self.model.forward(self.x, self.freqs_cis, self.cache)
        probabilities = torch.softmax(logits[-1], dim=-1)  # Convert last logits to probabilities

        # Get the top two probabilities.
        top_probs, _ = torch.topk(probabilities, 2, dim=-1)

        # Define high confidence criteria and rewards.
        high_confidence_threshold = 0.8  # Example threshold for high confidence
        medium_confidence_threshold = 0.5  # Example threshold for significant difference between top 1 and top 2
        print(f"top_probs: {top_probs[0].item()}")
        # Scenario 1: Reward if the top token's probability is above the high confidence threshold.
        if top_probs[0].item() > high_confidence_threshold:
            return 1.0  # High reward for high confidence in the top token.

        # Scenario 2: Reward if there's a significant difference between the top two tokens' probabilities.
        elif (top_probs[0] > medium_confidence_threshold):
            return 0.5  # Medium reward for significant confidence difference between top 1 and top 2 tokens.

        # Default: Low or no reward for low confidence predictions.
        return 0.0

    def __eq__(self, other):
        """
        Overrides the default implementation of equality comparison.
        Two TransformerState instances are considered equal if their sequences (self.x) are identical.
        
        Args:
            other (TransformerState): Another instance to compare against.
            
        Returns:
            bool: True if the instances are considered equal, False otherwise.
        """
        if not isinstance(other, TransformerState):
            # Don't attempt to compare against unrelated types.
            return NotImplemented

        # Compare sequences (self.x) for equality.
        return torch.equal(self.x, other.x)


class TransformerAction:
    def __init__(self, token_id):
        """
        Initializes a TransformerAction instance.
        
        Args:
            token_id (int): The identifier of the token this action represents.
        """
        self.token_id = token_id

    def __eq__(self, other):
        """
        Checks if another TransformerAction is equal to this one based on the token_id.
        
        Args:
            other (TransformerAction): The other action to compare against.
            
        Returns:
            bool: True if the actions are equal (have the same token_id), False otherwise.
        """
        if not isinstance(other, TransformerAction):
            # Ensure comparison is only between TransformerAction instances
            return NotImplemented

        return self.token_id == other.token_id
    
    def __lt__(self, other):
        return self.token_id < other.token_id
    
    def __gt__(self, other):
        return self.token_Id > other.token_id
    
    def __str__(self):
        return str(self.token_id)

    def __hash__(self):
        """
        Returns the hash of this action, allowing it to be used in sets and as dictionary keys.
        
        Returns:
            int: The hash of the action, based on its token_id.
        """
        return hash(self.token_id)

    def __repr__(self):
        return f"TransformerAction(token_id={self.token_id})"