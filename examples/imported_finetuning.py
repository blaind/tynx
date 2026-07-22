"""Fine-tune a slot-backed imported ONNX affine model."""

import tempfile
from pathlib import Path

import tynx


def main() -> None:
    fixture = Path(__file__).parent / "models" / "affine.onnx.hex"
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "affine.onnx"
        model_path.write_bytes(bytes.fromhex(fixture.read_text().strip()))
        model = tynx.load(
            model_path,
            trainable="auto",
            simplify=False,
            initializer_names={
                "constant1_out1": "head.weight",
                "constant2_out1": "head.bias",
            },
        )
        assert isinstance(model, tynx.ImportedModel)
        optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.05)
        input = tynx.Tensor([[-1.0], [1.0]])
        target = tynx.Tensor([[-5.0], [1.0]])

        initial_loss = tynx.nn.functional.mse_loss(model(input), target).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = tynx.nn.functional.mse_loss(model(input), target)
            loss.backward()
            optimizer.step()
        final_loss = tynx.nn.functional.mse_loss(model(input), target).item()

    print(f"imported ONNX fine-tuning: {initial_loss:.6f} -> {final_loss:.6f}")


if __name__ == "__main__":
    main()
