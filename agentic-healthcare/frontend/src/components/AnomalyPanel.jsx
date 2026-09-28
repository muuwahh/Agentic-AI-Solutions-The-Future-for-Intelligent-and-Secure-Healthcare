import { Activity, AlertTriangle, Watch } from 'lucide-react'

const wearableLabels = {
  heart_rate: 'Heart rate',
  spo2: 'SpO2',
  temperature: 'Temperature',
  motion: 'Motion',
}

export default function AnomalyPanel({ anomaly }) {
  const anomalous = anomaly.status === 'anomalous'
  return (
    <section className="panel anomaly-panel" aria-labelledby="anomaly-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Isolation Forest / wearable stream</p>
          <h3 id="anomaly-heading">Pattern monitor</h3>
        </div>
        <span className={`status-pill ${anomalous ? 'status-alert' : 'status-normal'}`}>
          {anomalous ? <AlertTriangle size={14} /> : <Activity size={14} />}
          {anomaly.status}
        </span>
      </div>
      <div className="anomaly-score">
        <div className="score-mark"><Watch size={21} /></div>
        <div>
          <span className="eyebrow">Anomaly score</span>
          <strong>{Number(anomaly.anomaly_score).toFixed(4)}</strong>
        </div>
        <span className="score-caption">relative to<br />training distribution</span>
      </div>
      <div className="wearable-list">
        {Object.entries(anomaly.input ?? {}).map(([key, value]) => (
          <div className="wearable-row" key={key}>
            <span>{wearableLabels[key] ?? key}</span>
            <strong>{value == null ? 'Missing' : Number(value).toFixed(1)}</strong>
          </div>
        ))}
      </div>
      <p className="panel-footnote">An unusual pattern does not imply illness. This detector is separate from the classifier.</p>
    </section>
  )
}