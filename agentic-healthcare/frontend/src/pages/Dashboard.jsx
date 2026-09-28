import { useEffect, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Database,
  HeartPulse,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Workflow,
} from 'lucide-react'
import AnomalyPanel from '../components/AnomalyPanel.jsx'
import EvaluationCharts from '../components/EvaluationCharts.jsx'
import MetricCard from '../components/MetricCard.jsx'
import PredictionPanel from '../components/PredictionPanel.jsx'
import { formatPercent, loadMlResults } from '../data/mlResults.js'

export default function Dashboard() {
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshedAt, setRefreshedAt] = useState(null)

  async function refreshResults() {
    setLoading(true)
    setError('')
    try {
      const nextResults = await loadMlResults()
      setResults(nextResults)
      setRefreshedAt(new Date())
    } catch (loadError) {
      setError(loadError.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refreshResults()
  }, [])

  const metrics = results?.metrics

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="Care Systems Lab home">
          <span className="brand-symbol"><HeartPulse size={19} strokeWidth={2.1} /></span>
          <span>CARE SYSTEMS <b>LAB</b></span>
        </a>
        <nav className="top-navigation" aria-label="Dashboard sections">
          <a className="nav-active" href="#overview">Overview</a>
          <a href="#evaluation">Model evaluation</a>
          <a href="#monitoring">Monitoring</a>
        </nav>
        <div className="topbar-meta"><span className="live-dot" />LOCAL RESEARCH RUN</div>
      </header>

      <main id="overview" className="dashboard-main">
        <section className="page-heading">
          <div>
            <div className="breadcrumb"><span>RESEARCH /</span> HEALTHCARE SYSTEMS</div>
            <h1>Intelligent care,<br /><em>under study.</em></h1>
            <p className="page-description">A transparent readout of locally trained models on synthetic demonstration data.</p>
          </div>
          <div className="run-controls">
            <div className="model-version"><span>MODEL VERSION</span><strong>{metrics?.model_version ?? '—'}</strong></div>
            <button className="refresh-button" type="button" onClick={refreshResults} disabled={loading}>
              <RefreshCw size={15} className={loading ? 'spin' : ''} />
              {loading ? 'Reading files' : 'Refresh results'}
            </button>
          </div>
        </section>

        <div className="dataset-ribbon">
          <div className="dataset-identity"><Database size={17} /><span>SYNTHETIC DEMONSTRATION DATA</span></div>
          <p>Arbitrary benchmark classes. No real patient records or clinical labels.</p>
          <span className="dataset-count">{metrics?.support ?? '—'} TEST ROWS</span>
        </div>

        {error && (
          <div className="load-error" role="alert">
            <ShieldCheck size={19} />
            <div><strong>Python results unavailable</strong><span>{error}</span></div>
          </div>
        )}

        <section className="metrics-section" aria-label="Held-out model metrics">
          <div className="section-heading compact-heading">
            <div><span className="section-number">01</span><h2>Held-out performance</h2></div>
            <span className="section-aside">TEST SPLIT · {metrics?.dataset ?? 'Awaiting Python output'}</span>
          </div>
          <div className="metric-grid">
            <MetricCard label="Accuracy" value={formatPercent(metrics?.accuracy)} detail="Correct test classifications" accent />
            <MetricCard label="Precision · macro" value={formatPercent(metrics?.precision)} detail="Unweighted class average" />
            <MetricCard label="Recall · macro" value={formatPercent(metrics?.recall)} detail="Unweighted class average" />
            <MetricCard label="Specificity · macro" value={formatPercent(metrics?.specificity)} detail="One-vs-rest class average" />
            <MetricCard label="F1 score · macro" value={formatPercent(metrics?.f1_score)} detail="Unweighted class average" />
            <MetricCard label="ROC–AUC · macro" value={formatPercent(metrics?.roc_auc)} detail="One-vs-rest class average" />
          </div>
        </section>

        <section id="evaluation" className="evaluation-section">
          <div className="section-heading">
            <div><span className="section-number">02</span><h2>Training &amp; evaluation</h2></div>
            <span className="section-aside">PYTORCH FNN <ArrowRight size={14} /> RFE · STANDARD SCALING</span>
          </div>
          <EvaluationCharts />
        </section>

        <section id="monitoring" className="monitoring-section">
          <div className="section-heading">
            <div><span className="section-number">03</span><h2>Model outputs</h2></div>
            <span className="section-aside">READ FROM PYTHON-GENERATED JSON</span>
          </div>
          {results && (
            <div className="output-grid">
              <PredictionPanel prediction={results.prediction} />
              <AnomalyPanel anomaly={results.anomaly} />
            </div>
          )}
        </section>

        <section className="workflow-strip">
          <div className="workflow-icon"><Workflow size={20} /></div>
          <div><span className="eyebrow">LOCAL ORCHESTRATION · {results?.workflow.workflow}</span><strong>{results?.workflow.steps.map((step, index) => <span key={step}>{index > 0 && <i>→</i>}{step}</span>)}</strong></div>
          <div className="workflow-outcomes"><span>FNN · {results?.workflow.patient_prediction.prediction}</span><span>WEARABLE · {results?.workflow.wearable_anomaly.status}</span></div>
          <div className="imaging-status"><ScanLine size={17} /><span>Medical imaging module: Not implemented</span></div>
          <div className="workflow-mark"><Activity size={19} /></div>
        </section>

        <footer className="footer">
          <span>RESEARCH PROTOTYPE · NOT A CLINICAL MEDICAL DEVICE</span>
          <span>{refreshedAt ? `FILES READ ${refreshedAt.toLocaleTimeString()}` : 'WAITING FOR MODEL OUTPUT'}</span>
        </footer>
      </main>
    </div>
  )
}