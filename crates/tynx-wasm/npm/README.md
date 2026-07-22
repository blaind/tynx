# tynxjs

Typed browser bindings for the Tynx ONNX runtime.

```sh
npm install tynxjs
```

```ts
import { Session } from "tynxjs";

const model = new Uint8Array(await (await fetch("model.onnx")).arrayBuffer());
const session = await Session.create(model, { backend: "webgpu" });
const output = await session.run(new Float32Array([1, 2, 3]), [3]);
session.free();
```

The portable default is `backend: "cpu"`. WebGPU requires browser support for WebGPU.
