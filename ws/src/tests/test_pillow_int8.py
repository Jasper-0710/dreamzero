import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None


@unittest.skipIf(torch is None, 'requires PyTorch')
class TestPillowInt8(unittest.TestCase):
    def test_weight_storage_and_bf16_forward(self):
        from torch import nn
        from ws.src.inference.pillow_int8 import Int8WeightLinear

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            linear = nn.Linear(64, 16).requires_grad_(False)
            linear.weight[0].zero_()
            x = torch.randn(2, 4, 64).to(torch.bfloat16)
        packed = Int8WeightLinear(linear).to(dtype=torch.bfloat16)
        self.assertEqual(packed.packed.dtype, torch.int8)
        self.assertEqual(packed.packed.numel(), linear.weight.numel())
        self.assertEqual(list(packed.parameters()), [])
        actual = packed(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        expected = linear.to(torch.bfloat16)(x)
        torch.testing.assert_close(actual[..., 0], expected[..., 0], rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)

    def test_only_transformer_block_linears_are_packed(self):
        from torch import nn
        from ws.src.inference.pillow_int8 import Int8WeightLinear, quantize_pillow_head

        head = nn.Module()
        head.model = nn.Module()
        head.model.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8))])
        head.model.action_decoder = nn.Linear(8, 14)
        head.text_encoder = nn.Module()
        head.text_encoder.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8, bias=False))])
        head.text_encoder.token_embedding = nn.Embedding(20, 8)
        head.image_encoder = nn.Linear(8, 8)
        head.eval()

        quantize_pillow_head(head)
        self.assertIsInstance(head.model.blocks[0][0], nn.LayerNorm)
        self.assertIsInstance(head.model.blocks[0][1], Int8WeightLinear)
        self.assertFalse(head.model.blocks[0][1].training)
        self.assertIsInstance(head.text_encoder.blocks[0][0], Int8WeightLinear)
        self.assertIsNone(head.text_encoder.blocks[0][0].bias)
        self.assertIsInstance(head.model.action_decoder, nn.Linear)
        self.assertIsInstance(head.text_encoder.token_embedding, nn.Embedding)
        self.assertIsInstance(head.image_encoder, nn.Linear)


if __name__ == '__main__':
    unittest.main()
