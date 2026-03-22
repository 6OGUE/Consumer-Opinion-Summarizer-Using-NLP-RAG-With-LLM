export type Stage =
  | "query"
  | "confirm_local"
  | "confirm_hosted"
  | "quantity"
  | "pipeline"
  | "done";

export interface RedditData {
  product: string;
  count: number;
  comments: unknown;
  sources: string[];
}

export interface ProcessedData {
  product: string;
  count: number;
  comments: Record<number, unknown>;
}

export interface PipelineStep {
  label: string;
  status: "idle" | "loading" | "done" | "error";
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}