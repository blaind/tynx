import { initialize, Session } from "tynxjs";
import type { Backend, SessionOptions } from "tynxjs";

async function example(model: ArrayBuffer, backend: Backend): Promise<void> {
  await initialize();
  const options: SessionOptions = { backend };
  const session = await Session.create(model, options);
  const output: Float32Array = await session.run([-1, 0, 1], [3]);
  console.log(session.inputs, session.outputs, output);
  session.free();
}

void example;
