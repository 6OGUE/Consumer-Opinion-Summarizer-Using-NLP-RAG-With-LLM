import type { PipelineStep } from '../types';

function ProgressBar({ steps }: { steps: PipelineStep[] }) {
  return (
    <div className="pipeline-steps">
      {steps.map((s, i) => (
        <div key={i} className={`step step--${s.status}`}>
          <div className="step-dot">
            {s.status === "loading" && <span className="spinner-sm" />}
            {s.status === "done" && <span className="check">✓</span>}
            {s.status === "error" && <span className="x-mark">✕</span>}
            {s.status === "idle" && <span className="idle-num">{i + 1}</span>}
          </div>
          <span className="step-label">{s.label}</span>
        </div>
      ))}
    </div>
  );
}

export default ProgressBar;