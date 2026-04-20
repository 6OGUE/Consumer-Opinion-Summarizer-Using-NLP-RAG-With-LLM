import type { PipelineStep } from './types';

export const BASE = "http://127.0.0.1:8000";

export const API_PATHS = {
  redditScrape: "/reddit_extract/scrape-reddit",
};

export const STEPS: PipelineStep[] = [
  { label: "Fetching Reddit data", status: "idle" },
  { label: "Removing duplicates", status: "idle" },
  { label: "Filtering comments", status: "idle" },
  { label: "Processing comments", status: "idle" },
  { label: "Scoring", status: "idle" },
  { label: "Final analysis", status: "idle" },
];