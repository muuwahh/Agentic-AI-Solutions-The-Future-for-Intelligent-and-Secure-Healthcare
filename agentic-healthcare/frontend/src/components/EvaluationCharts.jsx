const charts = [
  { file: 'loss_curve.png', title: 'Loss by epoch', note: 'Train and validation cross-entropy' },
  { file: 'accuracy_curve.png', title: 'Accuracy by epoch', note: 'Train and validation accuracy' },
  { file: 'confusion_matrix.png', title: 'Confusion matrix', note: 'Held-out test records' },
  { file: 'roc_curve.png', title: 'ROC curves', note: 'One-vs-rest, held-out test records' },
]

export default function EvaluationCharts() {
  return (
    <div className="chart-grid">
      {charts.map((chart, index) => (
        <figure className="chart-panel" key={chart.file} style={{ '--chart-order': index }}>
          <figcaption>
            <strong>{chart.title}</strong>
            <span>{chart.note}</span>
          </figcaption>
          <img src={`/${chart.file}`} alt={chart.title} />
        </figure>
      ))}
    </div>
  )
}