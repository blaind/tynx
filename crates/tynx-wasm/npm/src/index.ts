import initWasm, { Session as WasmSession } from "../wasm/tynx_wasm.js";
import type { InitInput } from "../wasm/tynx_wasm.js";

export type Backend = "cpu" | "webgpu";

export interface SessionOptions {
  /** Execution backend. CPU is the portable default. */
  backend?: Backend;
}

let initialization: Promise<void> | undefined;

/** Initialize the WebAssembly module. Calling this more than once is safe. */
export function initialize(
  moduleOrPath?: InitInput | Promise<InitInput>,
): Promise<void> {
  const input =
    moduleOrPath === undefined ? undefined : { module_or_path: moduleOrPath };
  initialization ??= initWasm(input).then(
    () => undefined,
    (error: unknown) => {
      initialization = undefined;
      throw error;
    },
  );
  return initialization;
}

/** A parsed ONNX model running in WebAssembly. */
export class Session {
  readonly #inner: WasmSession;

  private constructor(inner: WasmSession) {
    this.#inner = inner;
  }

  /** Initialize Tynx, parse an ONNX model, and select an execution backend. */
  static async create(
    model: Uint8Array | ArrayBuffer,
    options: SessionOptions = {},
  ): Promise<Session> {
    await initialize();
    const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
    const inner =
      options.backend === "webgpu"
        ? await WasmSession.withWebGpu(bytes)
        : new WasmSession(bytes);
    return new Session(inner);
  }

  /** Names of the model's declared inputs. */
  get inputs(): readonly string[] {
    return this.#inner.inputs;
  }

  /** Names of the model's declared outputs. */
  get outputs(): readonly string[] {
    return this.#inner.outputs;
  }

  /** Run a single-input, single-output f32 model. */
  run(
    input: Float32Array | readonly number[],
    shape: Uint32Array | readonly number[],
  ): Promise<Float32Array> {
    return this.#inner.run(Float32Array.from(input), Uint32Array.from(shape));
  }

  /** Release the model's WebAssembly resources. */
  free(): void {
    this.#inner.free();
  }
}
