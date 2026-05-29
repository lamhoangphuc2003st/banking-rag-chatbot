"use client";

import { Bot, Database, Send, ShieldCheck } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { sendChat } from "../lib/api";
import type { ChatMessage, SourceCitation } from "../types/chat";

type UiMessage = ChatMessage & {
  sources?: SourceCitation[];
};

const initialMessages: UiMessage[] = [
  {
    role: "assistant",
    content:
      "Xin chào. Tôi có thể tra cứu thông tin công khai của Vietcombank khi dữ liệu đã được crawl và index."
  }
];

export function ChatShell() {
  const [messages, setMessages] = useState<UiMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useMemo(() => crypto.randomUUID(), []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = input.trim();
    if (!content || isLoading) {
      return;
    }

    const nextMessages: UiMessage[] = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendChat(
        nextMessages.map(({ role, content }) => ({ role, content })),
        sessionId
      );
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources
        }
      ]);
    } catch {
      setError("Không gọi được API. Kiểm tra backend hoặc NEXT_PUBLIC_API_BASE_URL.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <Bot size={20} aria-hidden="true" />
          </div>
          <span>VCB RAG</span>
        </div>
        <div className="statusList" aria-label="Runtime status">
          <div className="statusItem">
            <Database size={16} aria-hidden="true" />
            <span>Qdrant + PostgreSQL</span>
          </div>
          <div className="statusItem">
            <ShieldCheck size={16} aria-hidden="true" />
            <span>Guardrails enabled</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <h1>Vietcombank RAG Chat</h1>
          <span className="pill">Public data only</span>
        </header>

        <section className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="bubble">{message.content}</div>
              {message.sources && message.sources.length > 0 ? (
                <div className="sources">
                  {message.sources.map((source) => (
                    <a
                      className="source"
                      href={source.source_url}
                      key={source.chunk_id}
                      rel="noreferrer"
                      target="_blank"
                    >
                      {source.title}
                    </a>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
        </section>

        {error ? <div className="error">{error}</div> : null}

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composerInner">
            <textarea
              aria-label="Nội dung câu hỏi"
              onChange={(event) => setInput(event.target.value)}
              placeholder="Nhập câu hỏi về sản phẩm, điều kiện, hồ sơ, biểu phí Vietcombank"
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
