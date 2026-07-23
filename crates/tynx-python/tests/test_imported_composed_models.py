"""Durable composed-model regressions for imported eager training."""

from pathlib import Path

import tynx

_CNN_MODEL = bytes.fromhex(
    "080a120a74796e782d74657374733aa0030a4d0a06696d616765730a0b636f6e762e776569"
    "6768740a09636f6e762e626961731208636f6e762e6f75741a04636f6e762204436f6e762a"
    "150a0c6b65726e656c5f736861706540014001a001070a200a08636f6e762e6f7574120872"
    "656c752e6f75741a0472656c75220452656c750a330a0872656c752e6f7574120866656174"
    "757265731a07666c617474656e2207466c617474656e2a0b0a04617869731801a001020a3a"
    "0a0866656174757265730a0b686561642e7765696768740a09686561642e62696173120a70"
    "726564696374696f6e1a0468656164220447656d6d12086d696e695f636e6e2a1d08010801"
    "080108011001420b636f6e762e7765696768744a040000003f2a15080110014209636f6e76"
    "2e626961734a04cdcccc3d2a25080408011001420b686561642e7765696768744a10000080"
    "3e000000bf0000403fcdcccc3d2a15080110014209686561642e626961734a04000000005a"
    "200a06696d6167657312160a14080112100a0208010a0208010a0208020a020802621c0a0a"
    "70726564696374696f6e120e0a0c080112080a0208010a02080142040a001012"
)
_RECOMMENDER_MODEL = bytes.fromhex(
    "08083aaa020a400a10656d62656464696e672e7765696768740a036964731207766563746f72"
    "731a09656d62656464696e6722064761746865722a0b0a04617869731800a001020a350a0776"
    "6563746f72730a0b686561642e7765696768740a09686561642e62696173120673636f726573"
    "1a0468656164220447656d6d12106d696e695f7265636f6d6d656e6465722a3a080408021001"
    "4210656d62656464696e672e7765696768744a200000803f00000000000000000000803f0000"
    "803f0000803f000080bf0000803f2a1d080208011001420b686561642e7765696768744a0800"
    "00003f000080be2a15080110014209686561642e626961734a04cdcccc3d5a110a0369647312"
    "0a0a08080712040a02080362180a0673636f726573120e0a0c080112080a0208030a02080142"
    "040a00100d"
)
_TRANSFORMER_MODEL = bytes.fromhex(
    "08083aeb060a530a16746f6b656e5f656d62656464696e672e7765696768740a09746f6b656e"
    "5f6964731208656d6265646465641a0f746f6b656e5f656d62656464696e6722064761746865"
    "722a0b0a04617869731800a001020a3c0a08656d6265646465640a1170726f6a656374696f6e"
    "2e776569676874120970726f6a65637465641a0a70726f6a656374696f6e22064d61744d756c"
    "0a2e0a0970726f6a65637465640a08656d6265646465641208726573696475616c1a08726573"
    "696475616c22034164640a710a08726573696475616c0a0b6e6f726d2e7765696768740a096e"
    "6f726d2e62696173120a6e6f726d616c697a65641a046e6f726d22124c617965724e6f726d61"
    "6c697a6174696f6e2a140a046178697318ffffffffffffffffff01a001022a110a07657073696c"
    "6f6e15acc52737a001010a490a0a6e6f726d616c697a65640a0b66697273742e696e64657812"
    "0b66697273745f72616e6b331a0c66697273742e67617468657222064761746865722a0b0a04"
    "617869731801a001020a380a0b66697273745f72616e6b330a0a66697273742e617865731205"
    "66697273741a0d66697273742e73717565657a65220753717565657a650a440a056669727374"
    "0a11636c61737369666965722e7765696768740a0f636c61737369666965722e626961731205"
    "6c6f6769741a0a636c6173736966696572220447656d6d12106d696e695f7472616e73666f72"
    "6d65722a400804080210014216746f6b656e5f656d62656464696e672e7765696768744a2000"
    "00803f00000000000000000000803f0000803f0000803f000080bf0000803f2a2b0802080210"
    "01421170726f6a656374696f6e2e7765696768744a100000003f000080be0000803e0000003f"
    "2a1b08021001420b6e6f726d2e7765696768744a080000803f0000803f2a190802100142096e"
    "6f726d2e626961734a0800000000000000002a1b08011007420b66697273742e696e6465784a"
    "0800000000000000002a1a08011007420a66697273742e617865734a0801000000000000002a"
    "230802080110014211636c61737369666965722e7765696768744a080000003f000000bf2a1b"
    "08011001420f636c61737369666965722e626961734a04000000005a1b0a09746f6b656e5f69"
    "6473120e0a0c080712080a0208010a02080262170a056c6f676974120e0a0c080112080a0208"
    "010a02080142040a001011"
)


def _train(
    model: tynx.ImportedModel,
    input: tynx.Tensor,
    target: tynx.Tensor,
    *,
    steps: int,
) -> tuple[float, float]:
    optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.03)
    initial = tynx.nn.functional.mse_loss(model(input), target).item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(model(input), target)
        loss.backward()
        optimizer.step()
    final = tynx.nn.functional.mse_loss(model(input), target).item()
    return initial, final


def test_imported_cnn_converges_through_convolution_and_dense_head(tmp_path: Path) -> None:
    path = tmp_path / "cnn.onnx"
    path.write_bytes(_CNN_MODEL)
    model = tynx.load(path, trainable="auto", simplify=False)
    report = model.require_trainable()

    initial, final = _train(
        model,
        tynx.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        tynx.Tensor([[1.5]]),
        steps=80,
    )

    assert report.warnings == []
    assert sorted(report.trainable_parameters) == [
        "conv.bias",
        "conv.weight",
        "head.bias",
        "head.weight",
    ]
    assert final < initial * 1e-3
    assert all(parameter.grad is not None for _, parameter in model.named_parameters())


def test_imported_recommender_trains_embedding_and_dense_head(tmp_path: Path) -> None:
    path = tmp_path / "recommender.onnx"
    path.write_bytes(_RECOMMENDER_MODEL)
    model = tynx.load(path, trainable="auto", simplify=False)
    report = model.require_trainable()

    initial, final = _train(
        model,
        tynx.Tensor([0, 1, 0], dtype="int64"),
        tynx.Tensor([[1.0], [-1.0], [1.0]]),
        steps=80,
    )

    assert report.warnings == []
    assert sorted(report.trainable_parameters) == [
        "embedding.weight",
        "head.bias",
        "head.weight",
    ]
    assert final < initial * 1e-3
    assert all(parameter.grad is not None for _, parameter in model.named_parameters())


def test_imported_transformer_slice_trains_all_parameter_families(tmp_path: Path) -> None:
    path = tmp_path / "transformer.onnx"
    path.write_bytes(_TRANSFORMER_MODEL)
    model = tynx.load(path, trainable="auto", simplify=False)
    report = model.require_trainable()

    initial, final = _train(
        model,
        tynx.Tensor([[0, 1]], dtype="int64"),
        tynx.Tensor([[2.0]]),
        steps=80,
    )

    assert report.warnings == []
    assert sorted(report.trainable_parameters) == [
        "classifier.bias",
        "classifier.weight",
        "norm.bias",
        "norm.weight",
        "projection.weight",
        "token_embedding.weight",
    ]
    assert final < initial * 1e-3
    assert all(parameter.grad is not None for _, parameter in model.named_parameters())
