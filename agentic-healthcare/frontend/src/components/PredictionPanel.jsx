import { ArrowDownRight, ArrowUpRight, CircleHelp } from 'lucide-react'

export default function PredictionPanel({ prediction }) {
  const features = prediction.important_features ?? []
  const maxChange = Math.max(...features.map((item) => Math.abs(item.probability_change)), 0.0001)

  return (
    <section className="panel prediction-panel" aria-labelledby="prediction-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">FNN / latest saved inference</p>
          <h3 id="prediction-heading">Demonstration classification</h3>
        </div>
        <span className="class-chip">{prediction.prediction}</span>
      </div>
      <div className="prediction-confidence">
        <span>Class probability</span>
        <strong>{(prediction.probability * 100).toFixed(1)}<small>%</small></strong>
      </div>
      <div className="probability-list" aria-label="Class probabilities">
        {Object.entries(prediction.class_probabilities ?? {}).map(([name, value]) => (
          <div className="probability-row" key={name}>
            <span>{name}</span>
            <div className="probability-track"><span style={{ width: `${value * 100}%` }} /></div>
            <span className="mono">{(value * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <div className="feature-heading">
        <span>Largest feature perturbations</span>
        <span className="muted-inline"><CircleHelp size={13} /> Not causal</span>
      </div>
      <div className="importance-list">
        {features.map((item) => {
          const positive = item.probability_change >= 0
          const ChangeIcon = positive ? ArrowUpRight : ArrowDownRight
          return (
            <div className="importance-row" key={item.feature}>
              <span>{item.feature.replaceAll('_', ' ')}</span>
              <div className="importance-track">
                <span
                  className={positive ? 'importance-positive' : 'importance-negative'}
                  style={{ width: `${(Math.abs(item.probability_change) / maxChange) * 100}%` }}
                />
              </div>
              <span className={`importance-value ${positive ? 'value-positive' : 'value-negative'}`}>
                <ChangeIcon size={14} />{item.probability_change > 0 ? '+' : ''}{item.probability_change.toFixed(3)}
              </span>
            </div>
          )
        })}
      </div>
      <p className="panel-footnote">{prediction.explanation_method}</p>
    </section>
  )
}