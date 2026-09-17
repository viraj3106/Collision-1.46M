/**
 * COLLISION Phase 100 — Production API Client.
 *
 * Provides a clean, centralized HTTP client connecting the frontend to
 * Phase 99 production API endpoints (/v1/ask, /health, /ready).
 */

export interface AskRequestPayload {
  question: string;
  mode?: string;
  include_sources?: boolean;
  include_claims?: boolean;
}

export interface SourceInfo {
  source_id: string;
  title: string;
  url: string;
  source_type: string;
  retrieval_score: number;
  relevance: number;
  snippet?: string;
}

export interface ClaimInfo {
  text: string;
  support_status: string;
  evidence_ids: string[];
}

export interface LatencyInfo {
  routing_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  verification_ms: number;
  total_ms: number;
}

export interface AskResponsePayload {
  answer: string;
  status: 'ANSWERED' | 'INSUFFICIENT_INFORMATION' | 'CONFLICT' | 'ERROR' | string;
  mode: string;
  confidence: number;
  sources: SourceInfo[];
  claims: ClaimInfo[];
  latency: LatencyInfo;
  metadata?: Record<string, any>;
}

export interface HealthCheckResponse {
  status: string;
  service: string;
  version: string;
  model: string;
}

export interface ReadyCheckResponse {
  status: string;
  service: string;
  checks: {
    model_checkpoint: boolean;
    tokenizer: boolean;
    local_index: boolean;
    web_provider: boolean;
  };
}

const API_BASE_URL = (
  import.meta.env.VITE_COLLISION_API_URL ||
  import.meta.env.VITE_API_URL ||
  'http://localhost:8000'
).replace(/\/$/, '');

export class CollisionApiClient {
  private baseUrl: string;
  private timeoutMs: number;

  constructor(baseUrl: string = API_BASE_URL, timeoutMs: number = 25000) {
    this.baseUrl = baseUrl;
    this.timeoutMs = timeoutMs;
  }

  public async ask(
    question: string,
    mode: string = 'AUTO',
    includeSources: boolean = true,
    includeClaims: boolean = true
  ): Promise<AskResponsePayload> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseUrl}/v1/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          question,
          mode,
          include_sources: includeSources,
          include_claims: includeClaims
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        const errMsg =
          errJson.error?.message ||
          errJson.detail?.message ||
          (typeof errJson.detail === 'string' ? errJson.detail : null) ||
          `Server returned HTTP ${response.status}`;

        return {
          answer: `⚠️ Error: ${errMsg}`,
          status: 'ERROR',
          mode: 'ERROR',
          confidence: 0,
          sources: [],
          claims: [],
          latency: {
            routing_ms: 0,
            retrieval_ms: 0,
            generation_ms: 0,
            verification_ms: 0,
            total_ms: 0
          },
          metadata: {
            http_status: response.status,
            error_code: errJson.error?.code || 'HTTP_ERROR'
          }
        };
      }

      const data: AskResponsePayload = await response.json();
      return data;
    } catch (err: any) {
      clearTimeout(timeoutId);
      const isTimeout = err.name === 'AbortError';
      const errMsg = isTimeout
        ? 'Request timed out after 25s. The model backend may be processing heavy load.'
        : `Network connection failed (${err.message || 'Server unreachable'}). Verify backend is running on port 8000.`;

      return {
        answer: `⚠️ Connection Error: ${errMsg}`,
        status: 'ERROR',
        mode: 'ERROR',
        confidence: 0,
        sources: [],
        claims: [],
        latency: {
          routing_ms: 0,
          retrieval_ms: 0,
          generation_ms: 0,
          verification_ms: 0,
          total_ms: 0
        },
        metadata: {
          is_timeout: isTimeout,
          client_error: err.message
        }
      };
    }
  }

  public async checkHealth(): Promise<HealthCheckResponse | null> {
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      if (res.ok) {
        return await res.json();
      }
      return null;
    } catch {
      return null;
    }
  }

  public async checkReady(): Promise<ReadyCheckResponse | null> {
    try {
      const res = await fetch(`${this.baseUrl}/ready`);
      if (res.ok) {
        return await res.json();
      }
      return null;
    } catch {
      return null;
    }
  }
}

export const collisionApi = new CollisionApiClient();
