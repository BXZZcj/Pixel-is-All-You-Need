import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator_benchmark import lora  # noqa: E402


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_k = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.other = nn.Linear(4, 4)

    def forward(self, x):
        return self.other(self.block(self.to_k(x)))


class LoraTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = Net()
        self.x = torch.randn(2, 4)

    def inject(self):
        return lora.inject(self.net, lora.default_matcher(("to_k", "block.2")), 2, 2)

    def test_untrained_lora_changes_nothing(self):
        before = self.net(self.x).detach().clone()
        self.assertEqual(self.inject(), ["block.2", "to_k"])
        self.assertTrue(torch.equal(self.net(self.x), before))

    def test_only_lora_params_train(self):
        self.inject()
        lora.freeze_base(self.net)
        self.assertEqual(sum(p.requires_grad for p in self.net.parameters()), 4)
        self.assertEqual(len(lora.trainable_parameters(self.net)), 4)
        base = self.net.other.weight.detach().clone()
        opt = torch.optim.SGD(lora.trainable_parameters(self.net), lr=0.1)
        for _ in range(3):
            self.net(self.x).pow(2).mean().backward()
            opt.step()
            opt.zero_grad()
        self.assertTrue(torch.equal(self.net.other.weight, base))
        self.assertGreater(self.net.to_k.up.weight.abs().sum().item(), 0)

    def test_scale_zero_is_base(self):
        base = self.net(self.x).detach().clone()
        self.inject()
        for _, m in lora.lora_modules(self.net):
            m.up.weight.data.fill_(0.5)
        self.assertFalse(torch.allclose(self.net(self.x), base))
        lora.set_scale(self.net, 0.0)
        self.assertTrue(torch.equal(self.net(self.x), base))

    def test_save_load_roundtrip(self):
        self.inject()
        for _, m in lora.lora_modules(self.net):
            m.up.weight.data.normal_()
        out = self.net(self.x).detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "l.safetensors"
            lora.save({"dit": self.net}, path, {"rank": 2})
            torch.manual_seed(0)
            fresh = Net()
            self.assertEqual(lora.load({"dit": fresh}, path, 2, 2), {"dit": 2})
        self.assertTrue(torch.allclose(fresh(self.x), out))

    def test_matcher_rejects_nothing_selected(self):
        with self.assertRaises(ValueError):
            lora.inject(self.net, lora.default_matcher(("missing",)), 2, 2)

    def test_float32_lora_on_bf16_base(self):
        self.net = self.net.to(torch.bfloat16)
        self.inject()
        out = self.net(self.x.to(torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(self.net.to_k.up.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
