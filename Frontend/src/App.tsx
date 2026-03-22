import { useState, useRef } from "react";
import "./App.css";

import type { Stage, RedditData, ProcessedData, PipelineStep, FinalInsightResponse } from './types';
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
  const [score, setScore] = useState<{ score: number } | null>(null);
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null);
  const [finalInsight, setFinalInsight] = useState<FinalInsightResponse | null>(null);
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
      setProcessedData(processed);
      setStep(3, "done");

      setStep(4, "loading");
      const scoreRes = await post<{ score: number }>("/score_finder/calc_score", processed);
      setScore(scoreRes);
      setStep(4, "done");

      setStep(5, "loading");
      const llmResult = await post<FinalInsightResponse>("/final_call/llm", processed);
      setFinalInsight(llmResult);
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
    setProcessedData(null);
    setFinalInsight(null);
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
            <div className="card-label">Enter Product</div>
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
            <div className="card-label">Confirm extracted product</div>
            <div className="extracted-box">
              <div className="extracted-label">Extracted Product</div>
              <div className="extracted-value">{extracted}</div>
            </div>
            <p style={{ fontSize: 13, color: "var(--muted)", marginBottom: 20 }}>
              Is this what you're looking for?
            </p>
            <div className="btn-row">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
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
            <div className="card-label">Data quantity</div>
            <div className="quantity-wrap">
              <div className="product-badge">
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>⬡</span>
                <span>{product}</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
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
            <div className="card-label">Pipeline</div>
            <ProgressBar steps={steps} />

            {error && <div className="error-msg">{error}</div>}

            {/* Results */}
            {stage === "done" && (
              <>
                <div className="divider" />

                <div className="result-section">
                  <div className="result-title">Final Analytics</div>
                  <p style={{ marginBottom: 16, color: "var(--muted)" }}>
                    Insights have been Prepared Successfully
                  </p>
                  <button
                    className="btn btn-primary"
                    onClick={() => setStage("insights")}
                    disabled={!finalInsight}
                  >
                    Go to Insights →
                  </button>
                </div>

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

      {/* Insights page */}
      {stage === "insights" && finalInsight && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-label">Insights</div>
          <div className="insights-header">
            <div className="score-section">
              <div className="score-circle" style={{
                background: score?.score != null && score.score > 50 ? 'rgba(62,255,163,0.1)' : 'rgba(255,94,94,0.1)',
                borderColor: score?.score != null && score.score > 50 ? 'var(--success)' : 'var(--error)'
              }}>
                <span>{score?.score != null ? `${score.score}%` : "N/A"}</span>
              </div>
              <div className="score-label">Overall score</div>
            </div>
            <div className="product-name">{processedData?.product.toUpperCase()}</div>
            <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px'}}>
            <button className="btn btn-chat" onClick={() => setShowChatbot(true)} style={{borderRadius: '50%', width: '64px', height: '64px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="2" x2="12" y2="5"/>
              <circle cx="12" cy="2" r="1" fill="currentColor" stroke="none"/>
              <rect x="3" y="5" width="18" height="14" rx="3" ry="3"/>
              <circle cx="9" cy="11" r="1.5" fill="currentColor" stroke="none"/>
              <circle cx="15" cy="11" r="1.5" fill="currentColor" stroke="none"/>
              <path d="M8 15 Q12 18 16 15"/>
            </svg>
  </button>
  <span style={{fontSize: '12px', fontWeight: 500}}>Chat</span>
</div>

          </div>
          <div className="insights-content">
            <h2>Overview</h2>
            <p>{finalInsight.overview}</p>

            <h3>Unique Features</h3>
            <ul>{finalInsight.unique_features.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <h3>Strengths</h3>
            <ul>{finalInsight.strengths.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <h3>Weaknesses</h3>
            <ul>{finalInsight.weaknesses.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <h3>Alternatives</h3>
            <ul>{finalInsight.alternatives.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <h3>Final Insight</h3>
            <p>{finalInsight.final_insight}</p>

            <div className="reset-row" style={{ marginTop: 20 }}>
              <button className="btn btn-ghost btn-sm" onClick={() => setStage("done")}>← Back</button>
              <button className="btn btn-ghost btn-sm" onClick={reset}>↩ Start over</button>
            </div>
          </div>
        </div>
      )}

      {/* Chatbot overlay — only rendered after pipeline is done */}
      {showChatbot && processedData && (
        <ChatBot
          processedData={processedData}
          onClose={() => setShowChatbot(false)}
        />
      )}
    </>
  );
}