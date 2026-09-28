async function readArtifact(filename) {
  const response = await fetch(`/${filename}`, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`${filename} is missing. Train the Python pipeline first.`)
  }
  return response.json()
}

export async function loadMlResults() {
  const [metrics, prediction, anomaly, workflow] = await Promise.all([
    readArtifact('model_results.json'),
    readArtifact('latest_prediction.json'),
    readArtifact('latest_anomaly.json'),
    readArtifact('latest_workflow.json'),
  ])
  return { metrics, prediction, anomaly, workflow }
}

export function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'N/A'
}