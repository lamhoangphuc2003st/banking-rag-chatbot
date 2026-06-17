"use client";

import { Bot, Send } from "lucide-react";
import { FormEvent, KeyboardEvent, useMemo, useState } from "react";
import { streamChat } from "../lib/api";
import type { ChatMessage, ClarificationOption, SourceCitation } from "../types/chat";

type UiMessage = ChatMessage & {
  sources?: SourceCitation[];
  clarificationOptions?: ClarificationOption[];
};

const initialMessages: UiMessage[] = [
  {
    role: "assistant",
    content: "Xin chào, bạn cần tra cứu thông tin gì về Vietcombank?"
  }
];

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function visibleSources(message: UiMessage): SourceCitation[] {
  if (!message.sources?.length) {
    return [];
  }

  const nonGroupSources = message.sources.filter((source) => !source.chunk_id.includes(":group:"));
  const displaySources = nonGroupSources.length > 0 ? nonGroupSources : message.sources;
  const seenUrls = new Set<string>();
  return displaySources.filter((source) => {
    if (seenUrls.has(source.source_url)) {
      return false;
    }
    seenUrls.add(source.source_url);
    return true;
  });
}

function sanitizeMessageContent(content: string): string {
  return content
    .split(/\r?\n/)
    .map((line) => sanitizeMessageLine(line))
    .filter((line, index, lines) => line.trim() || (index > 0 && index < lines.length - 1))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function sanitizeClarificationContent(content: string): string {
  return sanitizeMessageContent(content)
    .split(/\r?\n/)
    .map((line) => line.replace(/\s*Bạn có thể chọn một trong các mục liên quan sau:\s*$/i, ""))
    .filter((line) => !/^\s*\d{1,2}[.)]\s+.+(?:\s+\([^()]+\))?\s*$/.test(line))
    .filter((line, index, lines) => line.trim() || (index > 0 && index < lines.length - 1))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function sanitizeMessageLine(line: string): string {
  const markdownLinkOnly = line.match(
    /^\s*(?:[-*]\s*)?\[([^\]]+)\]\((?:https?:\/\/|www\.)[^\s)]+(?:\s+"[^"]*")?\)\s*[.!?]?\s*$/i
  );
  if (markdownLinkOnly && isDisposableLinkLabel(markdownLinkOnly[1])) {
    return "";
  }
  if (/^\s*(?:URL|Source|Link|Nguồn|Đường dẫn)\s*:\s*(?:https?:\/\/|www\.)\S+\s*$/i.test(line)) {
    return "";
  }
  if (isMissingInformationDisclaimer(line)) {
    return "";
  }

  const cleaned = line
    .replace(/\[([^\]]+)\]\((?:https?:\/\/|www\.)[^\s)]+(?:\s+"[^"]*")?\)/gi, "$1")
    .replace(/(?:https?:\/\/|www\.)[^\s)\]]+/gi, "")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/[ \t]{2,}/g, " ")
    .trimEnd();
  if (isMissingInformationDisclaimer(cleaned)) {
    return "";
  }
  return isDisposableLinkLabel(cleaned) ? "" : cleaned;
}

function normalizeText(text: string): string {
  const asciiText = text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d");
  return asciiText.replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
}

function isMissingInformationDisclaimer(text: string): boolean {
  const normalized = normalizeText(text);
  if (!normalized) {
    return false;
  }

  const missingMarkers = [
    "chua cung cap",
    "chua co thong tin",
    "chua tim thay",
    "khong co thong tin",
    "khong du thong tin",
    "khong tim thay",
    "nguon du lieu hien co chua",
    "nguon hien co chua"
  ];
  const adviceMarkers = [
    "ban can cung cap them",
    "ban vui long cung cap them",
    "kiem tra kenh chinh thuc",
    "kiem tra lai tren website",
    "de biet chi tiet hon"
  ];
  const missingFields = [
    "bieu phi",
    "chi tiet",
    "dieu kien",
    "doi tuong",
    "han muc",
    "ho so",
    "lai suat",
    "ngay hieu luc",
    "thu tuc",
    "yeu cau"
  ];

  const hasMissingMarker = missingMarkers.some((marker) => normalized.includes(marker));
  const hasAdviceMarker = adviceMarkers.some((marker) => normalized.includes(marker));
  const hasMissingField = missingFields.some((marker) => normalized.includes(marker));
  return (hasMissingMarker && (hasMissingField || hasAdviceMarker)) || (hasAdviceMarker && hasMissingField);
}

function isDisposableLinkLabel(label: string): boolean {
  return ["chi tiết", "xem chi tiết", "xem thêm", "tham khảo", "tham khảo thêm", "nguồn", "link"].includes(
    label.trim().replace(/^[-*:：.\s]+|[-*:：.\s]+$/g, "").toLowerCase()
  );
}

function clarificationOptionsFromMetadata(metadata: Record<string, unknown>): ClarificationOption[] {
  const directOptions = parseClarificationOptions(metadata.clarification_options);
  if (directOptions.length > 0) {
    return directOptions;
  }

  return parseClarificationOptions(metadata.retrieval_plan_clarification_options);
}

function parseClarificationOptions(value: unknown): ClarificationOption[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const options: ClarificationOption[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    if (typeof record.title !== "string" || !record.title.trim()) {
      continue;
    }

    options.push({
      title: record.title.trim(),
      type: typeof record.type === "string" ? record.type : "subject",
      url: typeof record.url === "string" ? record.url : null,
      product_type: typeof record.product_type === "string" ? record.product_type : null,
      category_title: typeof record.category_title === "string" ? record.category_title : null,
      parent_title: typeof record.parent_title === "string" ? record.parent_title : null
    });
  }
  return options;
}

function clarificationOptionLabel(option: ClarificationOption): string {
  if (option.type === "category") {
    return option.parent_title ? `${option.title} · nhóm con` : `${option.title} · nhóm`;
  }
  if (option.type === "product") {
    return `${option.title} · sản phẩm`;
  }
  return option.title;
}

export function ChatShell() {
  const [messages, setMessages] = useState<UiMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useMemo(() => createSessionId(), []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitMessage();
  }

  async function submitMessage(messageContent?: string, displayContent?: string) {
    const content = (messageContent ?? input).trim();
    if (!content || isLoading) {
      return;
    }
    const visibleContent = (displayContent ?? content).trim();

    const requestMessages: ChatMessage[] = [
      ...messages.map(({ role, content }) => ({ role, content })),
      { role: "user", content }
    ];
    const assistantIndex = requestMessages.length;
    const assistantMessage: UiMessage = { role: "assistant", content: "" };
    const visibleMessages: UiMessage[] = [
      ...messages,
      { role: "user", content: visibleContent },
      assistantMessage
    ];
    setMessages(visibleMessages);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      await streamChat(requestMessages, sessionId, {
        onToken: (token) => {
          setMessages((currentMessages) =>
            currentMessages.map((message, index) =>
              index === assistantIndex
                ? { ...message, content: `${message.content}${token}` }
                : message
            )
          );
        },
        onSources: (sources) => {
          setMessages((currentMessages) =>
            currentMessages.map((message, index) =>
              index === assistantIndex ? { ...message, sources } : message
            )
          );
        },
        onMetadata: (metadata) => {
          const clarificationOptions = clarificationOptionsFromMetadata(metadata);
          if (clarificationOptions.length === 0) {
            return;
          }
          setMessages((currentMessages) =>
            currentMessages.map((message, index) =>
              index === assistantIndex ? { ...message, clarificationOptions } : message
            )
          );
        }
      });
    } catch {
      setError("Không gọi được API. Kiểm tra backend hoặc NEXT_PUBLIC_API_BASE_URL.");
      setMessages(visibleMessages.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

    event.preventDefault();
    void submitMessage();
  }

  function handleClarificationClick(option: ClarificationOption, optionIndex: number) {
    void submitMessage(`Chọn ${optionIndex + 1}`, option.title);
  }

  return (
    <div className="shell">
      <main className="main">
        <header className="topbar">
          <div className="brand">
            <div className="brandMark">
              <Bot size={20} aria-hidden="true" />
            </div>
            <h1>Vietcombank RAG</h1>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {messages.map((message, index) => {
            const clarificationOptions = message.clarificationOptions ?? [];
            const displayContent =
              message.role === "assistant" && clarificationOptions.length > 0
                ? sanitizeClarificationContent(message.content)
                : message.role === "assistant"
                  ? sanitizeMessageContent(message.content)
                  : message.content;
            const sources = visibleSources(message);
            const canChooseOption =
              message.role === "assistant" &&
              clarificationOptions.length > 0 &&
              index === messages.length - 1 &&
              !isLoading;

            return (
              <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <div className="bubble">{displayContent}</div>
                {canChooseOption ? (
                  <div className="clarificationOptions" aria-label="Lựa chọn làm rõ">
                    {clarificationOptions.map((option, optionIndex) => (
                      <button
                        className="clarificationButton"
                        key={`${option.title}-${optionIndex}`}
                        onClick={() => handleClarificationClick(option, optionIndex)}
                        type="button"
                      >
                        {clarificationOptionLabel(option)}
                      </button>
                    ))}
                  </div>
                ) : null}
                {sources.length > 0 ? (
                  <div className="sources">
                    {sources.map((source) => (
                      <a
                        className="source"
                        href={source.source_url}
                        key={source.chunk_id}
                        rel="noreferrer"
                        target="_blank"
                        title={source.source_url}
                      >
                        {source.title}
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            );
          })}
        </section>

        {error ? <div className="error">{error}</div> : null}

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composerInner">
            <textarea
              aria-label="Nội dung câu hỏi"
              onKeyDown={handleComposerKeyDown}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Nhập câu hỏi về Vietcombank"
              value={input}
            />
            <button className="sendButton" disabled={isLoading || !input.trim()} title="Gửi" type="submit">
              <Send size={18} aria-hidden="true" />
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
