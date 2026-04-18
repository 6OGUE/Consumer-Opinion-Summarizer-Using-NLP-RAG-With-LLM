import { useState, useRef } from "react";
import "./App.css";

import type { Stage, RedditData, ProcessedData, PipelineStep, FinalInsightResponse, ScoreResponse, CommentResult, ChatMessage } from './types';
import { STEPS } from './constants';
import { post } from './utils';
import ProgressBar from './components/ProgressBar';
import ChatBot from './components/ChatBot';
import { exportInsightsToPDF } from './utils/pdfExport';

/* Visual-only helper component — no logic */
function ScoreRing({ score, isHigh }: { score: number; isHigh: boolean }) {
  const r = 46;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  const color = isHigh ? 'var(--cyan)' : 'var(--error)';
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" className="score-svg">
      <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="7" />
      <circle
        cx="60" cy="60" r={r} fill="none"
        stroke={color} strokeWidth="7"
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round" transform="rotate(-90 60 60)"
        style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1)' }}
      />
      <text x="60" y="65" textAnchor="middle" fill={color}
        fontSize="22" fontWeight="800" fontFamily="'Syne', sans-serif">
        {score}%
      </text>
    </svg>
  );
}

export default function App() {
  const [stage, setStage] = useState<Stage>("query");
  const [query, setQuery] = useState("");
  const [extracted, setExtracted] = useState("");
  const [product, setProduct] = useState("");
  const [quantity, setQuantity] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [steps, setSteps] = useState<PipelineStep[]>(STEPS.map((s) => ({ ...s })));
  const [score, setScore] = useState<ScoreResponse | null>(null);
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null);
  const [finalInsight, setFinalInsight] = useState<FinalInsightResponse | null>(null);
  const [sources, setSources] = useState<string[]>([]);
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
      
      // First classify the product to get category and aspects
      const classification = await post<{ product: string; category: string; aspects: string[] }>("/classify_product/classify_product", {
        extracted:product,
      });
      
      // Then scrape Reddit with the full data
      const reddit = await post<RedditData>("/reddit_extract/scrape-reddit", {
        product: classification.product,
        category: classification.category,
        aspects: classification.aspects,
        limit,
      });
      setSources(reddit.sources);
      setStep(0, "done");

      let payload = {
  product: reddit.product,
  category: reddit.category,     // ✅ added
  aspects: reddit.aspects,       // ✅ added
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
      const scoreRes = await post<ScoreResponse>("/score_finder/calc_score", processed);
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
    setSources([]);
    setShowChatbot(false);
    setSteps(STEPS.map((s) => ({ ...s })));
  }

  return (
    <>
      <div className="shell">

        {/* ── Stage: Query ── */}
        {stage === "query" && (
          <div className="query-stage">
            <div className="brand-eyebrow"></div>
            <h1 className="hero-title">Consumer Opinion<br />Summarizer</h1>
            <div className="card query-card">
              <div className="card-label">Enter Product</div>
              <div className="input-row">
                <input
                  className="input-field"
                  type="text"
                  placeholder="e.g. Is the iphone 13 worth purchasing?"
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
          </div>
        )}

        {/* ── Stage: Confirm Local ── */}
        {stage === "confirm_local" && (
          <div className="confirm-stage">
            <div className="page-eyebrow">Consumer Opinion Summarizer</div>
            <div className="card card--light">
              <div className="card-label">Confirm extracted product</div>
              <div className="extracted-box">
                <div className="extracted-label">Extracted Product</div>
                <div className="extracted-value">{extracted}</div>
              </div>
              <p className="confirm-hint">Is this what you're looking for?</p>
              <div className="btn-stack">
                <button className="btn btn-primary btn-full" onClick={handleConfirm}>
                  Yes, continue →
                </button>
                <button
                  className="btn btn-danger btn-full"
                  onClick={handleHosted}
                  disabled={loading}
                >
                  {loading ? <span className="spinner spinner--dark" /> : "✕  No, try again"}
                </button>
              </div>
              {error && <div className="error-msg">{error}</div>}
            </div>
            <div className="page-footer-label"></div>
          </div>
        )}

        {/* ── Stage: Confirm Hosted ── */}
        {stage === "confirm_hosted" && (
          <div className="confirm-stage">
            <div className="page-eyebrow">Consumer Opinion Summarizer</div>
            <div className="card card--light">
              <div className="card-label">Confirm extracted product</div>
              <div className="extracted-box">
                <div className="extracted-label">Extracted via hosted model</div>
                <div className="extracted-value">{extracted}</div>
              </div>
              <p className="confirm-hint">Does this look right?</p>
              <div className="btn-stack">
                <button className="btn btn-primary btn-full" onClick={handleConfirm}>
                  Yes, continue →
                </button>
                <button className="btn btn-danger btn-full" onClick={handleFinalReject}>
                  ✕  No — use my original query
                </button>
              </div>
            </div>
            <div className="page-footer-label"></div>
          </div>
        )}

        {/* ── Stage: Quantity ── */}
        {stage === "quantity" && (
          <div className="quantity-stage">
            <div className="brand-eyebrow--sm"></div>
            <div className="qty-badge">Data Quantity</div>
            <div className="qty-product-name">{product}</div>
            <div className="card qty-card">
              <div className="quantity-wrap">
                <p className="qty-question">How many Reddit comments should we fetch?</p>
                <input
                  className="input-field input-field--lg"
                  type="number"
                  placeholder="e.g. 100"
                  min={1}
                  value={quantity}
                  onChange={(e) => setQuantity(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleRunPipeline()}
                  autoFocus
                />
                <button
                  className="btn btn-primary btn-full"
                  onClick={handleRunPipeline}
                  disabled={!quantity}
                >
                  Run →
                </button>
                {error && <div className="error-msg">{error}</div>}
              </div>
            </div>
            <div className="page-footer-label">Estimated processing time: ~12 seconds</div>
          </div>
        )}

        {/* ── Stage: Pipeline ── */}
        {(stage === "pipeline" || stage === "done") && (
          <div className="pipeline-stage" ref={resultRef}>
            <div className="brand-eyebrow"></div>
            <div className="card pipeline-card">
              <div className="card-label" style={{ textAlign: 'center' }}></div>
              <ProgressBar steps={steps} />
              <div className="pipeline-divider" />
              <div className="pipeline-engine-label"></div>
            </div>

            {error && <div className="error-msg">{error}</div>}

            {stage === "done" && (
              <>
                <div className="result-section">
                  <p className="result-hint">Insights have been prepared successfully</p>
                  <button
                    className="btn btn-primary"
                    onClick={() => setStage("insights")}
                    disabled={!finalInsight}
                  >
                    Go to Insights →
                  </button>
                </div>
                <div className="reset-row">
                  <button className="btn btn-ghost btn-sm" onClick={reset}>↩ Start over</button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* ── Insights page ── */}
      {stage === "insights" && finalInsight && (
        <div className="insights-page">

          <div className="insights-topbar">
            <div className="insights-topbar-left">
              <div className="card-label">Insights</div>
              <div className="insights-product-name">{processedData?.product.toUpperCase()}</div>
            </div>
            <div className="insights-action-btns">
              <button className="icon-btn" title="Chat" onClick={() => setShowChatbot(true)}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </button>
              <button
                className="icon-btn"
                title="Download PDF"
                onClick={() => finalInsight && exportInsightsToPDF(finalInsight, processedData?.product || '', score?.overall_score || null)}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
              </button>
            </div>
          </div>

          <div className="score-row">
            {score && (
              <>
                <div className="score-main">
                  <ScoreRing score={score.overall_score} isHigh={score.overall_score > 50} />
                  <div className="score-meta">
                    <div className="score-meta-label">Overall Score</div>
                  </div>
                </div>
                <div className="score-aspects">
                  {Object.entries(score.aspect_scores).map(([aspect, value]) => (
                    <div key={aspect} className="score-aspect-item">
                      <ScoreRing score={value} isHigh={value > 50} />
                      <div className="score-aspect-label">{aspect}</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="insights-body">
            <h2>Overview</h2>
            <p>{finalInsight.overview}</p>

            <div className="divider" />
            <h3>Highlights</h3>
            <ul>{finalInsight.unique_features.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <div className="divider" />
            <h3>Strengths</h3>
            <ul>{finalInsight.strengths.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <div className="divider" />
            <h3>Weaknesses</h3>
            <ul>{finalInsight.weaknesses.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <div className="divider" />
            <h3>Alternatives</h3>
            <ul className="centered-list">{finalInsight.alternatives.map((item, idx) => <li key={idx}>{item}</li>)}</ul>

            <div className="divider" />
            <h2>Final Insight</h2>
            {[finalInsight.final_insight].map((item, idx) => <li key={idx}>{item}</li>)}

            {sources.length > 0 && (
              <>
                <div className="divider" />
                <h3>Sources</h3>
                <div className="sources-list">
                  {sources.map((source, idx) => (
                    <div key={idx} className="source-item">
                      <a href={source} target="_blank" rel="noopener noreferrer">{source}</a>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="reset-row" style={{ marginTop: 28 }}>
            <button className="btn btn-ghost btn-sm" onClick={() => setStage("done")}>← Back</button>
            <button className="btn btn-ghost btn-sm" onClick={reset}>↩ Start over</button>
          </div>
        </div>
      )}

      {/* Chatbot overlay */}
      {showChatbot && processedData && (
        <ChatBot
          processedData={processedData}
          onClose={() => setShowChatbot(false)}
        />
      )}
    </>
  );
}