import * as vscode from "vscode"
import { Package } from "../../../shared/package"

let apiChannel: vscode.OutputChannel | undefined
let rooChannel: vscode.OutputChannel | undefined

export function apiDebug(message: string, data?: any) {
	const ts = new Date().toISOString()
	const line = data !== undefined ? `${ts} ${message} ${safeStringify(data)}` : `${ts} ${message}`
	try {
		if (!apiChannel) apiChannel = vscode.window.createOutputChannel("Roo API Debug")
		apiChannel.appendLine(line)
	} catch {}

	// Also mirror into the main Roo Code output channel for discoverability
	try {
		if (!rooChannel) rooChannel = vscode.window.createOutputChannel(Package.outputChannel)
		rooChannel.appendLine(`[API Debug] ${line}`)
	} catch {}
}

function safeStringify(value: any): string {
	try {
		return typeof value === "string" ? value : JSON.stringify(value)
	} catch {
		return String(value)
	}
}
