import type { ChatMessage, ChatResponse, ChatStreamEvent, SourceCitation } from "../types/chat";

const API_BASE_URL = normalizeApiBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL);

export async function sendChat(
  messages: ChatMessage[],
  sessionId: string
): Promise<ChatResponse> {
  const response = await fetch(apiUrl("/v1/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      messages,
      session_id: sessionId
    })
  });

  if (!response.ok) {
    throw new Error(`API request failed with ${response.status}`);
  }

  return (await response.json()) as ChatResponse;
}

type StreamHandlers = {
  onToken: (token: string) => void;
  onSources: (sources: SourceCitation[]) => void;
  onMetadata?: (metadata: Record<string, unknown>) => void;
};

export async function streamChat(
  messages: ChatMessage[],
  sessionId: string,
  handlers: StreamHandlers
): Promise<void> {
  const response = await fetch(apiUrl("/v1/chat/stream"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      messages,
      session_id: sessionId
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`API stream request failed with ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const event = parseSseFrame(frame);
      if (!event) {
        continue;
      }
      if (event.type === "token") {
        handlers.onToken(event.content);
      }
      if (event.type === "sources") {
        handlers.onSources(event.sources);
      }
      if (event.type === "metadata") {
        handlers.onMetadata?.(event.metadata);
      }
      if (event.type === "done") {
        return;
      }
    }
  }
}

function normalizeApiBaseUrl(value: string | undefined): string {
  const configured = value?.trim();
  if (!configured) {
    return "/api/backend";
  }
  return configured.replace(/\/+$/, "");
}

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

function parseSseFrame(frame: string): ChatStreamEvent | null {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trim())
    .join("");

  if (!data) {
    return null;
  }

  return JSON.parse(data) as ChatStreamEvent;
}
