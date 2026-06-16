export type Role = "user" | "assistant" | "system";

export type ChatMessage = {
  role: Role;
  content: string;
};

export type SourceCitation = {
  chunk_id: string;
  title: string;
  source_url: string;
  section?: string | null;
  score?: number | null;
};

export type ClarificationOption = {
  title: string;
  type: string;
  url?: string | null;
  product_type?: string | null;
  category_title?: string | null;
  parent_title?: string | null;
};

export type ChatResponse = {
  answer: string;
  session_id: string;
  trace_id: string;
  sources: SourceCitation[];
  refusal: boolean;
  latency_ms: number;
  metadata: Record<string, unknown>;
};

export type ChatStreamEvent =
  | {
      type: "token";
      content: string;
    }
  | {
      type: "sources";
      sources: SourceCitation[];
    }
  | {
      type: "metadata";
      trace_id: string;
      refusal: boolean;
      latency_ms: number;
      metadata: Record<string, unknown>;
    }
  | {
      type: "done";
    };
