export default function MetricCard({ label, value, detail, accent = false }) {
  return (
    <article className={`metric-card${accent ? ' metric-card-accent' : ''}`}>
      <p className="eyebrow">{label}</p>
      <p className="metric-value">{value}</p>
      <p className="metric-detail">{detail}</p>
    </article>
  )
}