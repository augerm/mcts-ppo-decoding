import unittest
from unittest.mock import MagicMock, patch
import torch
from main import generate 

class TestGenerateFunction(unittest.TestCase):
    def setUp(self):
        # Mock the Transformer model
        self.model_mock = MagicMock()
        self.model_mock.args.vocab_size = 100
        self.model_mock.args.max_batch_size = 3
        self.model_mock.args.sliding_window = 1024
        self.model_mock.args.n_kv_heads = 8
        self.model_mock.args.head_dim = 64
        self.model_mock.device = 'cpu'
        self.model_mock.dtype = torch.float32
        self.model_mock.n_local_layers = 12
        self.model_mock.eval.return_value = self.model_mock

        # Setup the model mock to return a realistic tensor when forward is called
        self.model_mock.forward.side_effect = lambda input_ids, **kwargs: torch.rand((input_ids.shape[0], self.model_mock.args.vocab_size))

        # Mock the Tokenizer
        self.tokenizer_mock = MagicMock()
        self.tokenizer_mock.encode.side_effect = lambda prompt, bos: [1] + [min(ord(c), 99) for c in prompt]
        self.tokenizer_mock.decode.side_effect = lambda tokens: ''.join([chr(t) for t in tokens])

        # Patch the RotatingBufferCache class to return a mocked cache
        self.cache_patch = patch('mistral.cache.RotatingBufferCache', return_value=MagicMock())
        self.cache_patch.start()

        # If using MCTS, mock it as well
        self.mcts_mock = MagicMock()
        self.mcts_patch = patch('tree_search_decoding.mcts.MCTS', return_value=self.mcts_mock)
        self.mcts_patch.start()
        self.mcts_mock.search.return_value = torch.tensor([4])  # Mock search to return a specific token

    def tearDown(self):
        self.cache_patch.stop()
        self.mcts_patch.stop()
        self.cache_patch.stop()
        self.mcts_patch.stop()

    def test_generate_with_mcts(self):
        prompts = ["Test prompt"]
        max_tokens = 5
        temperature = 0.7

        # Call the generate function with MCTS decoding enabled
        generated_words, logprobs, detailed_logprobs = generate(
            prompts=prompts,
            model=self.model_mock,
            tokenizer=self.tokenizer_mock,
            max_tokens=max_tokens,
            temperature=temperature,
            use_mcts_decoding=True
        )

        # Verify MCTS was used
        self.mcts_mock.search.assert_called()
        self.assertGreaterEqual(len(generated_words), 1)
        self.assertGreaterEqual(len(logprobs[0]), 1)  # Assuming at least one token is generated

if __name__ == '__main__':
    unittest.main()
