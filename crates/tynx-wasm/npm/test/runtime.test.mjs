import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { initialize, Session } from "tynxjs";

const SIGN_MODEL_HEX =
  "0807120c6261636b656e642d746573743a3b0a0c0a017812017922045369676e1209746573745f7369676e5a0f0a0178120a0a08080112040a02080b620f0a0179120a0a08080112040a02080b42040a00100d";
const SIGN_MODEL = Uint8Array.from(
  SIGN_MODEL_HEX.matchAll(/../g),
  ([byte]) => Number.parseInt(byte, 16),
);

test("runs an ONNX model through the npm API", async () => {
  const wasm = await readFile(
    new URL("../wasm/tynx_wasm_bg.wasm", import.meta.url),
  );
  await initialize(wasm);

  const session = await Session.create(SIGN_MODEL);
  assert.deepEqual(session.inputs, ["x"]);
  assert.deepEqual(session.outputs, ["y"]);
  assert.deepEqual(
    Array.from(await session.run([-2, 0, 3], [3])),
    [-1, 0, 1],
  );
  session.free();
});
