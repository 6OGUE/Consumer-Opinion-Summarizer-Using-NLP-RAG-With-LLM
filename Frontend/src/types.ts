export type Stage =
  | "query"
  | "confirm_local"
  | "confirm_hosted"
  | "quantity"
  | "pipeline"
  | "done"
  | "insights";

export interface RedditData {
  product: string;
  category: string;        // ✅ added
  aspects: string[];       // ✅ added
  count: number;
  comments: unknown;
  sources: string[];
}

export interface CommentResult {
  summary: string;
  sentiments: Record<string, string>;   // ✅ added
  overall_sentiment: string;            // ✅ added
}

export interface ProcessedData {
  product: string;
  category: string;                    // ✅ added
  aspects: string[];                  // ✅ added
  count: number;
  comments: Record<number, CommentResult>; // ✅ fixed type
}

export interface FinalInsightResponse {
  overview: string;
  unique_features: string[];
  strengths: string[];
  weaknesses: string[];
  alternatives: string[];
  final_insight: string;
}

export interface ScoreResponse {
  overall_score: number;              // ✅ updated
  aspect_scores: Record<string, number>; // ✅ added
}

export interface PipelineStep {
  id: Key | null | undefined;
  label: string;
  status: "idle" | "loading" | "done" | "error";
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}