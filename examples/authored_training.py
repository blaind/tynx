"""Train a small authored eager model."""

import tynx


def main() -> None:
    tynx.manual_seed(7)
    model = tynx.nn.Sequential(
        tynx.nn.Linear(1, 8),
        tynx.nn.ReLU(),
        tynx.nn.Linear(8, 1),
    )
    optimizer = tynx.optim.Adam(model.parameters(), lr=0.03)
    input = tynx.Tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    target = tynx.Tensor([[-5.0], [-2.0], [1.0], [4.0], [7.0]])

    loss = tynx.nn.functional.mse_loss(model(input), target)
    initial_loss = loss.item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(model(input), target)
        loss.backward()
        optimizer.step()

    final_loss = tynx.nn.functional.mse_loss(model(input), target).item()
    print(f"authored eager training: {initial_loss:.6f} -> {final_loss:.6f}")


if __name__ == "__main__":
    main()
