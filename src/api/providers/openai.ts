import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI, { AzureOpenAI } from "openai"
import { Agent, type Dispatcher, setGlobalDispatcher, fetch as undiciFetch } from "undici"
import axios from "axios"

import {
	type ModelInfo,
	azureOpenAiDefaultApiVersion,
	openAiModelInfoSaneDefaults,
	DEEP_SEEK_DEFAULT_TEMPERATURE,
	OPENAI_AZURE_AI_INFERENCE_PATH,
} from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { XmlMatcher } from "../../utils/xml-matcher"

import { convertToOpenAiMessages } from "../transform/openai-format"
import { convertToR1Format } from "../transform/r1-format"
import { convertToSimpleMessages } from "../transform/simple-format"
import { ApiStream, ApiStreamUsageChunk } from "../transform/stream"
import { getModelParams } from "../transform/model-params"

import { DEFAULT_HEADERS } from "./constants"
import { BaseProvider } from "./base-provider"
import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"
import { getApiRequestTimeout } from "./utils/timeout-config"
import { handleOpenAIError } from "./utils/openai-error-handler"
import { apiDebug } from "./utils/api-debug"
import * as vscode from "vscode"

// TODO: Rename this to OpenAICompatibleHandler. Also, I think the
// `OpenAINativeHandler` can subclass from this, since it's obviously
// compatible with the OpenAI API. We can also rename it to `OpenAIHandler`.
export class OpenAiHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: OpenAI
	private readonly providerName = "OpenAI"
	private static dispatcherCache = new Map<number, Dispatcher>()
	private static streamingDispatcherCache = new Map<number, Dispatcher>()
	private activeDispatcher?: Dispatcher
	private defaultDispatcher?: Dispatcher
	private streamingDispatcher?: Dispatcher

	private static getTimeoutConfig(timeout: number) {
		const cacheKey = timeout > 0 ? timeout : 0
		let dispatcher = this.dispatcherCache.get(cacheKey)
		if (!dispatcher) {
			const undiciTimeout = cacheKey === 0 ? 0 : cacheKey
			dispatcher = new Agent({
				headersTimeout: undiciTimeout,
				bodyTimeout: undiciTimeout,
				connectTimeout: undiciTimeout,
			})
			this.dispatcherCache.set(cacheKey, dispatcher)
		}

		// Streaming dispatcher: disable idle body timeout between bytes
		let streamingDispatcher = this.streamingDispatcherCache.get(cacheKey)
		if (!streamingDispatcher) {
			const undiciTimeout = cacheKey === 0 ? 0 : cacheKey
			streamingDispatcher = new Agent({
				headersTimeout: undiciTimeout,
				bodyTimeout: 0,
				connectTimeout: undiciTimeout,
			})
			this.streamingDispatcherCache.set(cacheKey, streamingDispatcher)
		}

		try {
			const payload = {
				inputTimeout: timeout,
				cacheKey,
				undiciConfigured: {
					headersTimeout: (dispatcher as any).headersTimeout,
					bodyTimeout: (dispatcher as any).bodyTimeout,
					connectTimeout: (dispatcher as any).connectTimeout,
				},
			}
			console.debug("[openai] getTimeoutConfig", payload)
			apiDebug("[openai] getTimeoutConfig", payload)
		} catch {}

		return {
			timeout,
			fetchOptions: { dispatcher },
			streamingFetchOptions: { dispatcher: streamingDispatcher },
		}
	}

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options

		const baseURL = this.options.openAiBaseUrl ?? "https://api.openai.com/v1"
		const apiKey = this.options.openAiApiKey ?? "not-provided"
		const isAzureAiInference = this._isAzureAiInference(this.options.openAiBaseUrl)
		const urlHost = this._getUrlHost(this.options.openAiBaseUrl)
		const isAzureOpenAi = urlHost === "azure.com" || urlHost.endsWith(".azure.com") || options.openAiUseAzure

		const headers = {
			...DEFAULT_HEADERS,
			...(this.options.openAiHeaders || {}),
		}

		const timeout = getApiRequestTimeout()
		const timeoutConfig = OpenAiHandler.getTimeoutConfig(timeout)
		const configuredDispatcher = (timeoutConfig.fetchOptions as any)?.dispatcher
		this.defaultDispatcher = configuredDispatcher
		this.streamingDispatcher = (timeoutConfig as any)?.streamingFetchOptions?.dispatcher
		this.activeDispatcher = this.defaultDispatcher
		// Ensure Node's global fetch uses our dispatcher (Node 20 global fetch ignores per-call dispatcher)
		try {
			if (configuredDispatcher) {
				setGlobalDispatcher(configuredDispatcher as Dispatcher)
				apiDebug("[openai] setGlobalDispatcher applied", {})
			}
		} catch {}
		try {
			const payload = {
				baseURL,
				isAzureAiInference,
				isAzureOpenAi,
				timeout: timeoutConfig.timeout,
				fetchOptions: Boolean(timeoutConfig.fetchOptions),
			}
			console.debug("[openai] client init", payload)
			apiDebug("[openai] client init", payload)
		} catch {}

		// Constructor visibility
		try {
			apiDebug("[openai] constructor", {
				baseURL,
				modelId: this.options.openAiModelId,
				streaming: this.options.openAiStreamingEnabled ?? true,
			})
		} catch {}

		// Debug fetch wrapper to trace aborts and durations and force our dispatcher
		const debugFetch: typeof fetch = async (input: any, init?: any) => {
			const started = Date.now()
			const url = typeof input === "string" ? input : (input && input.url) || String(input)
			const method = (init && init.method) || (input && input.method) || "GET"
			let aborted = false
			const signal: AbortSignal | undefined = init?.signal || (input && input.signal)
			// Always inject our dispatcher unless caller overrides explicitly
			const mergedInit = { ...(init || {}), dispatcher: (init as any)?.dispatcher ?? this.activeDispatcher }
			const onAbort = () => {
				aborted = true
				try {
					apiDebug("[openai] fetch abort event", { url, method, elapsedMs: Date.now() - started })
				} catch {}
			}
			try {
				signal?.addEventListener("abort", onAbort, { once: true })
			} catch {}
			try {
				let hdrs: any = undefined
				try {
					const h = (init as any)?.headers
					if (h) {
						if (h instanceof Headers) {
							hdrs = {}
							h.forEach((v, k) => (hdrs[k] = v))
						} else if (Array.isArray(h)) {
							hdrs = Object.fromEntries(h)
						} else if (typeof h === "object") {
							hdrs = h
						}
					}
				} catch {}
				let bodyInfo: any = undefined
				try {
					const b: any = (init as any)?.body
					if (typeof b === "string") bodyInfo = { type: "string", length: b.length, preview: b.slice(0, 200) }
					else if (b && typeof b === "object" && typeof b.text === "function") {
						const t = await b.text()
						bodyInfo = { type: "body-like", length: t.length, preview: t.slice(0, 200) }
					}
				} catch {}
				apiDebug("[openai] fetch start", { method, url, headers: hdrs, body: bodyInfo })
			} catch {}
			try {
				// Use undici's fetch directly to ensure dispatcher/timeouts are honored
				const res = await undiciFetch(input as any, mergedInit as any)
				try {
					apiDebug("[openai] fetch response", {
						url,
						status: res.status,
						elapsedMs: Date.now() - started,
						aborted,
						headers: {
							"content-type": res.headers.get("content-type"),
							"transfer-encoding": res.headers.get("transfer-encoding"),
							connection: res.headers.get("connection"),
						},
					})
				} catch {}
				return res
			} catch (e) {
				try {
					apiDebug("[openai] fetch error", {
						url,
						method,
						elapsedMs: Date.now() - started,
						aborted,
						message: e instanceof Error ? e.message : String(e),
						stack: e instanceof Error ? e.stack || "" : undefined,
						cause: e instanceof Error && (e as any).cause ? String((e as any).cause) : undefined,
					})
				} catch {}
				throw e
			} finally {
				try {
					signal?.removeEventListener("abort", onAbort)
				} catch {}
			}
		}

		if (isAzureAiInference) {
			// Azure AI Inference Service (e.g., for DeepSeek) uses a different path structure
			this.client = new OpenAI({
				baseURL,
				apiKey,
				defaultHeaders: headers,
				defaultQuery: { "api-version": this.options.azureApiVersion || "2024-05-01-preview" },
				timeout: timeoutConfig.timeout,
				fetchOptions: {
					...(timeoutConfig.fetchOptions || {}),
					dispatcher: (timeoutConfig.fetchOptions as any)?.dispatcher,
				},
				fetch: debugFetch as any,
			})
		} else if (isAzureOpenAi) {
			// Azure API shape slightly differs from the core API shape:
			// https://github.com/openai/openai-node?tab=readme-ov-file#microsoft-azure-openai
			this.client = new AzureOpenAI({
				baseURL,
				apiKey,
				apiVersion: this.options.azureApiVersion || azureOpenAiDefaultApiVersion,
				defaultHeaders: headers,
				timeout: timeoutConfig.timeout,
				fetchOptions: timeoutConfig.fetchOptions,
				fetch: debugFetch as any,
			})
		} else {
			this.client = new OpenAI({
				baseURL,
				apiKey,
				defaultHeaders: headers,
				timeout: timeoutConfig.timeout,
				fetchOptions: timeoutConfig.fetchOptions,
				fetch: debugFetch as any,
			})
		}
	}

	override async *createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
		metadata?: ApiHandlerCreateMessageMetadata & {
			thinking?: {
				enabled: boolean
				maxTokens?: number
				maxThinkingTokens?: number
			}
		},
	): ApiStream {
		// Entry log to guarantee visibility in Roo Code output
		try {
			const currentTimeout = getApiRequestTimeout()
			const payload = {
				modelId: this.options.apiModelId,
				baseUrl: this.options.openAiBaseUrl,
				streaming: this.options.openAiStreamingEnabled ?? true,
				configuredTimeoutMs: currentTimeout,
			}
			console.debug("[openai] createMessage enter", payload)
			apiDebug("[openai] createMessage enter", payload)
			if (process.env.NODE_ENV === "development") {
				// Dev-only toast to confirm code path is hit
				vscode.window.showInformationMessage("[Roo API Debug] OpenAI createMessage started")
			}
		} catch {}

		const { info: modelInfo, reasoning } = this.getModel()
		const modelUrl = this.options.openAiBaseUrl ?? ""
		const modelId = this.options.openAiModelId ?? ""
		const enabledR1Format = this.options.openAiR1FormatEnabled ?? false
		const enabledLegacyFormat = this.options.openAiLegacyFormat ?? false
		const isAzureAiInference = this._isAzureAiInference(modelUrl)
		const deepseekReasoner = modelId.includes("deepseek-reasoner") || enabledR1Format
		const ark = modelUrl.includes(".volces.com")

		if (modelId.includes("o1") || modelId.includes("o3") || modelId.includes("o4")) {
			yield* this.handleO3FamilyMessage(modelId, systemPrompt, messages)
			return
		}

		if (this.options.openAiStreamingEnabled ?? true) {
			let systemMessage: OpenAI.Chat.ChatCompletionSystemMessageParam = {
				role: "system",
				content: systemPrompt,
			}

			let convertedMessages

			if (deepseekReasoner) {
				convertedMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
			} else if (ark || enabledLegacyFormat) {
				convertedMessages = [systemMessage, ...convertToSimpleMessages(messages)]
			} else {
				if (modelInfo.supportsPromptCache) {
					systemMessage = {
						role: "system",
						content: [
							{
								type: "text",
								text: systemPrompt,
								// @ts-ignore-next-line
								cache_control: { type: "ephemeral" },
							},
						],
					}
				}

				convertedMessages = [systemMessage, ...convertToOpenAiMessages(messages)]

				if (modelInfo.supportsPromptCache) {
					// Note: the following logic is copied from openrouter:
					// Add cache_control to the last two user messages
					// (note: this works because we only ever add one user message at a time, but if we added multiple we'd need to mark the user message before the last assistant message)
					const lastTwoUserMessages = convertedMessages.filter((msg) => msg.role === "user").slice(-2)

					lastTwoUserMessages.forEach((msg) => {
						if (typeof msg.content === "string") {
							msg.content = [{ type: "text", text: msg.content }]
						}

						if (Array.isArray(msg.content)) {
							// NOTE: this is fine since env details will always be added at the end. but if it weren't there, and the user added a image_url type message, it would pop a text part before it and then move it after to the end.
							let lastTextPart = msg.content.filter((part) => part.type === "text").pop()

							if (!lastTextPart) {
								lastTextPart = { type: "text", text: "..." }
								msg.content.push(lastTextPart)
							}

							// @ts-ignore-next-line
							lastTextPart["cache_control"] = { type: "ephemeral" }
						}
					})
				}
			}

			const isGrokXAI = this._isGrokXAI(this.options.openAiBaseUrl)

			const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
				model: modelId,
				temperature: this.options.modelTemperature ?? (deepseekReasoner ? DEEP_SEEK_DEFAULT_TEMPERATURE : 0),
				messages: convertedMessages,
				stream: true as const,
				...(isGrokXAI ? {} : { stream_options: { include_usage: true } }),
				...(reasoning && reasoning),
			}

			// Add max_tokens if needed
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			// Diagnostics: summarize request for compatibility issues
			try {
				const hasStreamOptions = Boolean((requestOptions as any).stream_options)
				const msgStats = {
					count: Array.isArray(requestOptions.messages) ? requestOptions.messages.length : 0,
					roles: Array.isArray(requestOptions.messages)
						? (requestOptions.messages as any[]).reduce(
								(acc, m) => {
									acc[m.role] = (acc[m.role] || 0) + 1
									return acc
								},
								{} as Record<string, number>,
							)
						: {},
					hasCacheControl: Array.isArray(requestOptions.messages)
						? (requestOptions.messages as any[]).some(
								(m) => Array.isArray(m.content) && m.content.some((p: any) => p?.cache_control),
							)
						: false,
				}
				apiDebug("[openai] requestOptions summary", {
					model: requestOptions.model,
					hasStreamOptions,
					msgStats,
				})
			} catch {}

			// Switch to streaming dispatcher (no body idle timeout) for the duration of this streaming request
			const prevDispatcher = this.activeDispatcher
			this.activeDispatcher = this.streamingDispatcher ?? prevDispatcher
			try {
				apiDebug("[openai] streaming dispatcher active", {
					bodyTimeout: 0,
					note: "idle body timeout disabled for streaming",
				})
			} catch {}

			let stream
			try {
				const payload = { provider: this.providerName }
				console.debug("[openai] create chat.completions (streaming) request", payload)
				apiDebug("[openai] create chat.completions (streaming) request", payload)
				stream = await this.client.chat.completions.create(
					requestOptions,
					isAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
				)
			} catch (error) {
				try {
					const payload = {
						provider: this.providerName,
						error: error instanceof Error ? { name: error.name, message: error.message } : String(error),
					}
					console.debug("[openai] error creating streaming request", payload)
					apiDebug("[openai] error creating streaming request", payload)
				} catch {}
				throw handleOpenAIError(error, this.providerName)
			}

			const matcher = new XmlMatcher(
				"think",
				(chunk) =>
					({
						type: chunk.matched ? "reasoning" : "text",
						text: chunk.data,
					}) as const,
			)

			let lastUsage
			const startedAt = Date.now()
			let lastChunkAt = 0
			let chunkCount = 0
			const heartbeat = setInterval(() => {
				try {
					const now = Date.now()
					apiDebug("[openai] stream heartbeat", {
						elapsedMs: now - startedAt,
						sinceLastChunkMs: lastChunkAt ? now - lastChunkAt : null,
						chunks: chunkCount,
					})
				} catch {}
			}, 30_000)
			const nearTenMinWarn = setTimeout(() => {
				try {
					apiDebug("[openai] nearing 10m window", {
						elapsedMs: Date.now() - startedAt,
						chunks: chunkCount,
						sinceLastChunkMs: lastChunkAt ? Date.now() - lastChunkAt : null,
					})
				} catch {}
			}, 590_000)

			try {
				console.debug("[openai] streaming started")
				apiDebug("[openai] streaming started")
				let firstChunk = true
				const noChunkTicker = setInterval(() => {
					if (!firstChunk) return
					try {
						apiDebug("[openai] no chunks yet", { elapsedMs: Date.now() - startedAt })
					} catch {}
				}, 60_000)
				for await (const chunk of stream) {
					chunkCount++
					lastChunkAt = Date.now()
					const delta = chunk.choices[0]?.delta ?? {}
					if (firstChunk) {
						firstChunk = false
						try {
							apiDebug("[openai] first chunk", { elapsedMs: Date.now() - startedAt })
						} catch {}
					}

					if (delta.content) {
						for (const part of matcher.update(delta.content)) {
							yield part
						}
					}

					if ("reasoning_content" in delta && (delta as any).reasoning_content) {
						yield {
							type: "reasoning",
							text: ((delta as any).reasoning_content as string | undefined) || "",
						}
					}
					if (chunk.usage) {
						lastUsage = chunk.usage
					}
				}
				clearInterval(noChunkTicker)

				for (const part of matcher.final()) {
					yield part
				}

				if (lastUsage) {
					yield this.processUsageMetrics(lastUsage, modelInfo)
				}
			} catch (err) {
				try {
					const e: any = err
					apiDebug("[openai] streaming error", {
						name: e?.name,
						code: e?.code,
						status: e?.status,
						message: e instanceof Error ? e.message : String(e),
						stack: e instanceof Error ? e.stack || "" : undefined,
						elapsedMs: Date.now() - startedAt,
						chunks: chunkCount,
						lastChunkAgoMs: lastChunkAt ? Date.now() - lastChunkAt : null,
					})
				} catch {}
				throw err
			} finally {
				clearInterval(heartbeat)
				clearTimeout(nearTenMinWarn)
				try {
					apiDebug("[openai] streaming finished", {
						elapsedMs: Date.now() - startedAt,
						chunks: chunkCount,
						lastChunkAgoMs: lastChunkAt ? Date.now() - lastChunkAt : null,
					})
				} catch {}
				// Restore the default dispatcher after streaming completes
				this.activeDispatcher = prevDispatcher
			}
		} else {
			// o1 for instance doesnt support streaming, non-1 temp, or system prompt
			const systemMessage: OpenAI.Chat.ChatCompletionUserMessageParam = {
				role: "user",
				content: systemPrompt,
			}

			const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
				model: modelId,
				messages: deepseekReasoner
					? convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
					: enabledLegacyFormat
						? [systemMessage, ...convertToSimpleMessages(messages)]
						: [systemMessage, ...convertToOpenAiMessages(messages)],
			}

			// Add max_tokens if needed
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			let response
			try {
				const payload = { provider: this.providerName }
				console.debug("[openai] create chat.completions (non-streaming) request", payload)
				apiDebug("[openai] create chat.completions (non-streaming) request", payload)
				response = await this.client.chat.completions.create(
					requestOptions,
					this._isAzureAiInference(modelUrl) ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
				)
			} catch (error) {
				try {
					const payload = {
						provider: this.providerName,
						error: error instanceof Error ? { name: error.name, message: error.message } : String(error),
					}
					console.debug("[openai] error creating non-streaming request", payload)
					apiDebug("[openai] error creating non-streaming request", payload)
				} catch {}
				throw handleOpenAIError(error, this.providerName)
			}

			yield {
				type: "text",
				text: response.choices[0]?.message.content || "",
			}

			yield this.processUsageMetrics(response.usage, modelInfo)
		}
	}

	protected processUsageMetrics(usage: any, _modelInfo?: ModelInfo): ApiStreamUsageChunk {
		return {
			type: "usage",
			inputTokens: usage?.prompt_tokens || 0,
			outputTokens: usage?.completion_tokens || 0,
			cacheWriteTokens: usage?.cache_creation_input_tokens || undefined,
			cacheReadTokens: usage?.cache_read_input_tokens || undefined,
		}
	}

	override getModel() {
		const id = this.options.openAiModelId ?? ""
		const info = this.options.openAiCustomModelInfo ?? openAiModelInfoSaneDefaults
		const params = getModelParams({ format: "openai", modelId: id, model: info, settings: this.options })
		return { id, info, ...params }
	}

	async completePrompt(prompt: string): Promise<string> {
		try {
			apiDebug("[openai] completePrompt enter", {
				modelId: this.options.openAiModelId,
				baseUrl: this.options.openAiBaseUrl,
				promptPreview: prompt.slice(0, 20),
			})
			const modelConfig = this.getModel()
			const isAzureAiInference = this._isAzureAiInference(this.options.openAiBaseUrl)
			const model = this.getModel()
			const modelInfo = model.info

			const isGrokXAI = this._isGrokXAI(this.options.openAiBaseUrl)

			const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
				model: model.id,
				messages: [{ role: "user", content: prompt }],
			}

			// Add max_tokens if needed
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			let response
			try {
				response = await this.client.chat.completions.create(
					requestOptions,
					isAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
				)
			} catch (error) {
				throw handleOpenAIError(error, this.providerName)
			}

			return response.choices[0]?.message.content || ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`${this.providerName} completion error: ${error.message}`)
			}

			throw error
		}
	}

	private async *handleO3FamilyMessage(
		modelId: string,
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
	): ApiStream {
		const modelInfo = this.getModel().info
		const methodIsAzureAiInference = this._isAzureAiInference(this.options.openAiBaseUrl)

		if (this.options.openAiStreamingEnabled ?? true) {
			const isGrokXAI = this._isGrokXAI(this.options.openAiBaseUrl)

			const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
				model: modelId,
				messages: [
					{
						role: "developer",
						content: `Formatting re-enabled\n${systemPrompt}`,
					},
					...convertToOpenAiMessages(messages),
				],
				stream: true,
				...(isGrokXAI ? {} : { stream_options: { include_usage: true } }),
				reasoning_effort: modelInfo.reasoningEffort as "low" | "medium" | "high" | undefined,
				temperature: undefined,
			}

			// O3 family models do not support the deprecated max_tokens parameter
			// but they do support max_completion_tokens (the modern OpenAI parameter)
			// This allows O3 models to limit response length when includeMaxTokens is enabled
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			let stream
			try {
				stream = await this.client.chat.completions.create(
					requestOptions,
					methodIsAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
				)
			} catch (error) {
				throw handleOpenAIError(error, this.providerName)
			}

			yield* this.handleStreamResponse(stream)
		} else {
			const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
				model: modelId,
				messages: [
					{
						role: "developer",
						content: `Formatting re-enabled\n${systemPrompt}`,
					},
					...convertToOpenAiMessages(messages),
				],
				reasoning_effort: modelInfo.reasoningEffort as "low" | "medium" | "high" | undefined,
				temperature: undefined,
			}

			// O3 family models do not support the deprecated max_tokens parameter
			// but they do support max_completion_tokens (the modern OpenAI parameter)
			// This allows O3 models to limit response length when includeMaxTokens is enabled
			this.addMaxTokensIfNeeded(requestOptions, modelInfo)

			let response
			try {
				response = await this.client.chat.completions.create(
					requestOptions,
					methodIsAzureAiInference ? { path: OPENAI_AZURE_AI_INFERENCE_PATH } : {},
				)
			} catch (error) {
				throw handleOpenAIError(error, this.providerName)
			}

			yield {
				type: "text",
				text: response.choices[0]?.message.content || "",
			}
			yield this.processUsageMetrics(response.usage)
		}
	}

	private async *handleStreamResponse(stream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>): ApiStream {
		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	private _getUrlHost(baseUrl?: string): string {
		try {
			return new URL(baseUrl ?? "").host
		} catch (error) {
			return ""
		}
	}

	private _isGrokXAI(baseUrl?: string): boolean {
		const urlHost = this._getUrlHost(baseUrl)
		return urlHost.includes("x.ai")
	}

	private _isAzureAiInference(baseUrl?: string): boolean {
		const urlHost = this._getUrlHost(baseUrl)
		return urlHost.endsWith(".services.ai.azure.com")
	}

	/**
	 * Adds max_completion_tokens to the request body if needed based on provider configuration
	 * Note: max_tokens is deprecated in favor of max_completion_tokens as per OpenAI documentation
	 * O3 family models handle max_tokens separately in handleO3FamilyMessage
	 */
	protected addMaxTokensIfNeeded(
		requestOptions:
			| OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming
			| OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
		modelInfo: ModelInfo,
	): void {
		// Only add max_completion_tokens if includeMaxTokens is true
		if (this.options.includeMaxTokens === true) {
			// Use user-configured modelMaxTokens if available, otherwise fall back to model's default maxTokens
			// Using max_completion_tokens as max_tokens is deprecated
			requestOptions.max_completion_tokens = this.options.modelMaxTokens || modelInfo.maxTokens
		}
	}
}

export async function getOpenAiModels(baseUrl?: string, apiKey?: string, openAiHeaders?: Record<string, string>) {
	try {
		if (!baseUrl) {
			return []
		}

		// Trim whitespace from baseUrl to handle cases where users accidentally include spaces
		const trimmedBaseUrl = baseUrl.trim()

		if (!URL.canParse(trimmedBaseUrl)) {
			return []
		}

		const config: Record<string, any> = {}
		const headers: Record<string, string> = {
			...DEFAULT_HEADERS,
			...(openAiHeaders || {}),
		}

		if (apiKey) {
			headers["Authorization"] = `Bearer ${apiKey}`
		}

		if (Object.keys(headers).length > 0) {
			config["headers"] = headers
		}

		const response = await axios.get(`${trimmedBaseUrl}/models`, config)
		const modelsArray = response.data?.data?.map((model: any) => model.id) || []
		return [...new Set<string>(modelsArray)]
	} catch (error) {
		return []
	}
}
