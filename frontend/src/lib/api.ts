/**
 * API Service Client for Scientific Contradiction Finder.
 * 
 * Communicates with backend ASGI server via Next.js proxy route /api/backend
 * or direct local server fallback.
 */

import {
  AnalysisResponse,
  BenchmarkClassifyResponse,
  BenchmarkPairsResponse,
  FindingPair,
  HealthResponse,
  Prediction,
  PresetsResponse,
} from "@/types/scientific";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== "undefined" ? "/api/backend" : "http://127.0.0.1:8000/api");

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(message: string, status: number = 500, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { cache: "no-store" });
    if (!res.ok) {
      throw new ApiError(`Health check failed with status ${res.status}`, res.status);
    }
    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      "Unable to connect to research backend server. Ensure backend is running.",
      0,
      err instanceof Error ? err.message : String(err)
    );
  }
}

export async function fetchPresets(): Promise<string[]> {
  try {
    const res = await fetch(`${BASE_URL}/presets`, { cache: "no-store" });
    if (!res.ok) {
      throw new ApiError(`Failed to fetch presets (${res.status})`, res.status);
    }
    const data: PresetsResponse = await res.json();
    return data.presets || [];
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    return [
      "Do randomized controlled trials show that low-carbohydrate diets produce greater weight loss than low-fat diets, and does the difference persist over time?",
      "Does moderate alcohol consumption reduce all-cause mortality, or can the apparent protective association be explained by study design and confounding?",
      "Does vitamin D supplementation prevent cardiovascular events in generally healthy adults, and how do randomized trials compare with observational findings?",
    ];
  }
}

export async function analyzeQuery(
  query: string,
  options: { limit?: number; useGroq?: boolean } = {}
): Promise<AnalysisResponse> {
  const { limit = 10, useGroq = false } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout for live retrieval + extraction

  try {
    const res = await fetch(`${BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        limit,
        use_groq_interpretation: useGroq,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errJson = await res.json().catch(() => null);
      throw new ApiError(
        errJson?.error || `Analysis failed with HTTP status ${res.status}`,
        res.status,
        errJson?.detail
      );
    }

    return await res.json();
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError(
        "Request timed out while retrieving literature across scholarly providers.",
        408
      );
    }
    throw new ApiError(
      "Network connection error communicating with research pipeline.",
      500,
      err instanceof Error ? err.message : String(err)
    );
  }
}

export async function fetchBenchmarkPairs(): Promise<BenchmarkPairsResponse> {
  try {
    const res = await fetch(`${BASE_URL}/benchmark/pairs`, { cache: "no-store" });
    if (!res.ok) {
      throw new ApiError(`Failed to load benchmark pairs (${res.status})`, res.status);
    }
    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      "Failed to load CAS-Con150 benchmark pairs.",
      500,
      err instanceof Error ? err.message : String(err)
    );
  }
}

export async function classifyPair(pair: FindingPair): Promise<Prediction> {
  try {
    const res = await fetch(`${BASE_URL}/benchmark/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pair }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => null);
      throw new ApiError(
        errJson?.error || `Classification failed (${res.status})`,
        res.status,
        errJson?.detail
      );
    }

    const data: BenchmarkClassifyResponse = await res.json();
    return data.prediction;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      "Failed to classify finding pair.",
      500,
      err instanceof Error ? err.message : String(err)
    );
  }
}
