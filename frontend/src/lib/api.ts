import { FullAnalysisResponse, SystemHealth, ModelInfo, ExperimentMetric } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function analyzeNews(formData: FormData): Promise<FullAnalysisResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed with status ${res.status}`);
  }
  return res.json();
}

export async function getAnalysis(id: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/v1/analysis/${id}`);
  if (!res.ok) throw new Error("Failed to fetch analysis");
  return res.json();
}

export async function getAnalysisHistory(): Promise<any[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/history`);
  if (!res.ok) throw new Error("Failed to fetch analysis history");
  return res.json();
}

export async function getSystemHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`);
  if (!res.ok) throw new Error("Failed to check system health");
  return res.json();
}

export async function getModelsInfo(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/models`);
  if (!res.ok) throw new Error("Failed to fetch models");
  return res.json();
}

export async function getEvaluationMetrics(): Promise<{
  experiments: Record<string, ExperimentMetric>;
  markdown_report: string;
}> {
  const res = await fetch(`${API_BASE_URL}/api/v1/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function submitFeedback(
  analysisId: string,
  type: string,
  text?: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId, feedback_type: type, feedback_text: text }),
  });
  return res.json();
}
