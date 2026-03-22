import { useState, useRef } from "react";
import "./App.css";

import type { Stage, RedditData, ProcessedData, PipelineStep } from './types';
import { STEPS } from './constants';
import { post } from './utils';
import ProgressBar from './components/ProgressBar';
import ChatBot from './components/ChatBot';







export default function App() {
  const [stage, setStage] = useState<Stage>("query");
  const [query, setQuery] = useState("");
  const [extracted, setExtracted] = useState("");
  const [product, setProduct] = useState("");
  const [quantity, setQuantity] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [steps, setSteps] = useState<PipelineStep[]>(STEPS.map((s) => ({ ...s })));
  const [sources, setSources] = useState<string[]>([]);
  const [score, setScore] = useState<unknown>(null);
  const [finalResult, setFinalResult] = useState<ProcessedData | null>(null);
  const [showChatbot, setShowChatbot] = useState(false);
  const resultRef = useRef<HTMLDivElement>(null);

  function setStep(index: number, status: PipelineStep["status"]) {
    setSteps((prev) =>
      prev.map((s, i) => (i === index ? { ...s, status } : s))
    );
  }

  async function handleGo() {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await post<{ extracted: string }>("/product_extract_local/local", {
        query: query.trim(),
      });
      setExtracted(res.extracted);
      setStage("confirm_local");
    } catch {
      setError("Local extraction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  async function handleHosted() {
    setLoading(true);
    setError("");
    try {
      const res = await post<{ extracted: string }>("/product_extract_hosted/llm", {
        query: query.trim(),
      });
      setExtracted(res.extracted);
      setStage("confirm_hosted");
    } catch {
      setError("Hosted extraction failed. Falling back to your original query.");
      setProduct(query.trim());
      setStage("quantity");
    } finally {
      setLoading(false);
    }
  }

  function handleConfirm() {
    setProduct(extracted);
    setStage("quantity");
  }

  function handleFinalReject() {
    setProduct(query.trim());
    setStage("quantity");
  }

  async function handleRunPipeline() {
    const limit = parseInt(quantity, 10);
    if (!limit || limit < 1) {
      setError("Please enter a valid number.");
      return;
    }
    setStage("pipeline");
    setError("");
    const fresh = STEPS.map((s) => ({ ...s }));
    setSteps(fresh);

    try {
      setStep(0, "loading");
      const reddit = await post<RedditData>("/reddit_extract/scrape-reddit", {
        product,
        limit,
      });
      setSources(reddit.sources);
      setStep(0, "done");

      let payload: { product: string; count: number; comments: unknown } = {
        product: reddit.product,
        count: reddit.count,
        comments: reddit.comments,
      };

      setStep(1, "loading");
      payload = await post("/remove_duplicates/deduplicate", payload);
      setStep(1, "done");

      setStep(2, "loading");
      payload = await post("/cleanup_comments/filter-comments", payload);
      setStep(2, "done");

      setStep(3, "loading");
      const processed = await post<ProcessedData>("/process_comments/process_comments", payload);
      setStep(3, "done");

      setStep(4, "loading");
      const scoreRes = await post("/score_finder/calc_score", processed);
      setScore(scoreRes);
      setStep(4, "done");

      setStep(5, "loading");
      await new Promise(r => setTimeout(r, 400));
      setFinalResult(processed);
      setStep(5, "done");

      setStage("done");
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Pipeline error";
      setError(msg);
      setSteps((prev) =>
        prev.map((s) => (s.status === "loading" ? { ...s, status: "error" } : s))
      );
    }
  }

  function reset() {
    setStage("query");
    setQuery("");
    setExtracted("");
    setProduct("");
    setQuantity("");
    setError("");
    setScore(null);
    setFinalResult(null);
    setSources([]);
    setShowChatbot(false);
    setSteps(STEPS.map((s) => ({ ...s })));
  }

  return (
    <>
      <div className="shell">
        {/* Header */}
        <div className="header">
          <div className="header-tag">Product Intelligence</div>
          <h1>Consumer<br /><span></span> Opinion Summarizer</h1>
        </div>

        {/* ── Stage: Query ── */}
        {stage === "query" && (
          <div className="card">
            <div className="card-label">01 — Enter your product query</div>
            <div className="input-row">
              <input
                className="input-field"
                type="text"
                placeholder="e.g. Is the iphone 14 worth purchasing?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGo()}
                autoFocus
              />
              <button
                className="btn btn-primary"
                onClick={handleGo}
                disabled={loading || !query.trim()}
              >
                {loading ? <span className="spinner" /> : "Go →"}
              </button>
            </div>
            {error && <div className="error-msg">{error}</div>}
          </div>
        )}

        {/* ── Stage: Confirm Local ── */}
        {stage === "confirm_local" && (
          <div className="card">
            <div className="card-label">02 — Confirm extracted product</div>
            <div className="extracted-box">
              <div className="extracted-label">Extracted via local model</div>
              <div className="extracted-value">{extracted}</div>
            </div>
            <p style={{ fontSize: 13, color: "var(--muted)", marginBottom: 20 }}>
              Does this match what you're looking for?
            </p>
            <div className="btn-row">
              <button className="btn btn-primary btn-sm" onClick={handleConfirm}>
                ✓ Yes, continue
              </button>
              <button
                className="btn btn-danger btn-sm"
                onClick={handleHosted}
                disabled={loading}
              >
                {loading ? <span className="spinner" /> : "✕ No, try again"}
              </button>
            </div>
            {error && <div className="error-msg">{error}</div>}
          </div>
        )}

        {/* ── Stage: Confirm Hosted ── */}
        {stage === "confirm_hosted" && (
          <div className="card">
            <div className="card-label">02 — Confirm extracted product</div>
            <div className="extracted-box">
              <div className="extracted-label">Extracted via hosted model</div>
              <div className="extracted-value">{extracted}</div>
            </div>
            <p style={{ fontSize: 13, color: "var(--muted)", marginBottom: 20 }}>
              Does this look right?
            </p>
            <div className="btn-row">
              <button className="btn btn-primary btn-sm" onClick={handleConfirm}>
                ✓ Yes, continue
              </button>
              <button
                className="btn btn-danger btn-sm"
                onClick={handleFinalReject}
              >
                ✕ No — use my original query
              </button>
            </div>
          </div>
        )}

        {/* ── Stage: Quantity ── */}
        {stage === "quantity" && (
          <div className="card">
            <div className="card-label">03 — Data quantity</div>
            <div className="quantity-wrap">
              <div className="product-badge">
                <span>⬡</span>
                <span>{product}</span>
              </div>
              <p style={{ fontSize: 13, color: "var(--muted)" }}>
                How many Reddit comments should we fetch?
              </p>
              <div className="input-row">
                <input
                  className="input-field"
                  type="number"
                  placeholder="e.g. 100"
                  min={1}
                  value={quantity}
                  onChange={(e) => setQuantity(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleRunPipeline()}
                  autoFocus
                />
                <button
                  className="btn btn-primary"
                  onClick={handleRunPipeline}
                  disabled={!quantity}
                >
                  Run →
                </button>
              </div>
              {error && <div className="error-msg">{error}</div>}
            </div>
          </div>
        )}

        {/* ── Stage: Pipeline ── */}
        {(stage === "pipeline" || stage === "done") && (
          <div className="card" ref={resultRef}>
            <div className="card-label">04 — Pipeline</div>
            <ProgressBar steps={steps} />

            {error && <div className="error-msg">{error}</div>}

            {/* Results */}
            {stage === "done" && (
              <>
                <div className="divider" />

                {sources.length > 0 && (
                  <div className="result-section">
                    <div className="result-title">Sources ({sources.length})</div>
                    <div className="sources-list">
                      {sources.map((s, i) => (
                        <div key={i} className="source-item" title={s}>{s}</div>
                      ))}
                    </div>
                  </div>
                )}

                {score !== null && (
                  <div className="result-section">
                    <div className="result-title">Score</div>
                    <div className="result-json">
                      {JSON.stringify(score, null, 2)}
                    </div>
                  </div>
                )}

                {finalResult !== null && (
                  <div className="result-section">
                    <div className="result-title">Final Result</div>
                    <div className="result-json">
                      {JSON.stringify(finalResult, null, 2)}
                    </div>
                  </div>
                )}

                {/* Chat CTA */}
                {finalResult && (
                  <div className="chat-cta">
                    <div className="chat-cta-text">
                      <strong>Have questions?</strong> Ask our AI about the analysis.
                    </div>
                    <button
                      className="btn btn-chat"
                      onClick={() => setShowChatbot(true)}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                      </svg>
                      Chat about {finalResult.product}
                    </button>
                  </div>
                )}

                <div className="reset-row">
                  <button className="btn btn-ghost btn-sm" onClick={reset}>
                    ↩ Start over
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Chatbot overlay — only rendered after pipeline is done */}
      {showChatbot && finalResult && (
        <ChatBot
          processedData={finalResult}
          onClose={() => setShowChatbot(false)}
        />
      )}
    </>
  );
}