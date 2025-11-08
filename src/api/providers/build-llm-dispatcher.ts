import { Agent, type Dispatcher } from "undici"

export function buildLlmDispatcher(timeoutMs: number): Dispatcher {
	return new Agent({
		headersTimeout: timeoutMs + 10_000,
		bodyTimeout: timeoutMs + 10_000,
		connectTimeout: timeoutMs + 10_000,
	})
}
