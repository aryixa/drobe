/**
 * ML Pipeline API Service
 * Connects React frontend to the Python ML pipeline API
 */

export interface RecommendationRequest {
  query: string;
  max_outfits?: number;
  user_preferences?: Record<string, number>;
}

export interface RecommendationResponse {
  success: boolean;
  query: string;
  recommendation?: {
    items: string[];
    score: number;
    style_score: number;
    color_score: number;
    pattern_score: number;
    formality_score: number;
  };
  alternatives: Array<{
    items: string[];
    score: number;
    explanation: any;
  }>;
  explanation?: any;
  processing_time: number;
  metadata: Record<string, any>;
}

export interface ImageAnalysisRequest {
  image_path: string;
  analysis_type?: 'colors' | 'type' | 'pattern' | 'all';
}

export interface ImageAnalysisResponse {
  success: boolean;
  image_path: string;
  colors?: Array<{
    category: string;
    rgb: [number, number, number];
    percentage: number;
    confidence: number;
  }>;
  type_classification?: {
    primary_type: string;
    sub_type?: string;
    confidence: number;
    alternatives: Array<{
      type: string;
      confidence: number;
    }>;
  };
  pattern_detection?: {
    pattern_type: string;
    confidence: number;
    characteristics: Record<string, number>;
    evidence: Record<string, number>;
  };
  processing_time: number;
  errors: string[];
}

export interface ContextParseResponse {
  original_query: string;
  occasion?: string;
  season?: string;
  weather?: string;
  time_of_day?: string;
  style_level?: string;
  colors: string[];
  patterns: string[];
  clothing_types: string[];
  keywords: string[];
  confidence: number;
  parsing_errors: string[];
}

export interface BatchRecommendationRequest {
  queries: string[];
  user_preferences?: Record<string, number>;
}

export interface BatchRecommendationResponse {
  success: boolean;
  results: Array<{
    query: string;
    success: boolean;
    recommendation?: {
      items: string[];
      score: number;
    };
    explanation?: any;
    processing_time: number;
    error?: string;
  }>;
  total_processed: number;
  total_time: number;
  errors: string[];
}

export interface HealthResponse {
  status: string;
  components: Record<string, string>;
  capabilities: string[];
}

class MLApiService {
  private baseUrl: string;
  private isHealthy: boolean = false;

  constructor(baseUrl: string = 'http://127.0.0.1:8000') {
    this.baseUrl = baseUrl;
    this.checkHealth();
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async checkHealth(): Promise<HealthResponse> {
    try {
      const health = await this.request<HealthResponse>('/health');
      this.isHealthy = health.status === 'healthy';
      return health;
    } catch (error) {
      this.isHealthy = false;
      throw error;
    }
  }

  isApiHealthy(): boolean {
    return this.isHealthy;
  }

  /**
   * Get outfit recommendation based on natural language query
   */
  async getRecommendation(request: RecommendationRequest): Promise<RecommendationResponse> {
    if (!this.isHealthy) {
      await this.checkHealth();
    }

    return this.request<RecommendationResponse>('/recommend', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Analyze clothing image for colors, type, and patterns
   */
  async analyzeImage(request: ImageAnalysisRequest): Promise<ImageAnalysisResponse> {
    if (!this.isHealthy) {
      await this.checkHealth();
    }

    return this.request<ImageAnalysisResponse>('/analyze-image', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Parse natural language query into structured context
   */
  async parseContext(query: string): Promise<ContextParseResponse> {
    if (!this.isHealthy) {
      await this.checkHealth();
    }

    return this.request<ContextParseResponse>(`/context-parse?query=${encodeURIComponent(query)}`);
  }

  /**
   * Process multiple recommendation queries
   */
  async batchRecommend(request: BatchRecommendationRequest): Promise<BatchRecommendationResponse> {
    if (!this.isHealthy) {
      await this.checkHealth();
    }

    return this.request<BatchRecommendationResponse>('/batch-recommend', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get system statistics and performance metrics
   */
  async getStats(): Promise<any> {
    if (!this.isHealthy) {
      await this.checkHealth();
    }

    return this.request<any>('/stats');
  }

  /**
   * Get recommendation with error handling
   */
  async getRecommendationSafe(query: string, maxOutfits: number = 5): Promise<{
    success: boolean;
    data?: RecommendationResponse;
    error?: string;
  }> {
    try {
      const data = await this.getRecommendation({
        query,
        max_outfits: maxOutfits,
      });
      
      return { success: true, data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * Analyze image with error handling
   */
  async analyzeImageSafe(
    imagePath: string, 
    analysisType: 'colors' | 'type' | 'pattern' | 'all' = 'all'
  ): Promise<{
    success: boolean;
    data?: ImageAnalysisResponse;
    error?: string;
  }> {
    try {
      const data = await this.analyzeImage({
        image_path: imagePath,
        analysis_type: analysisType,
      });
      
      return { success: true, data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * Parse context with error handling
   */
  async parseContextSafe(query: string): Promise<{
    success: boolean;
    data?: ContextParseResponse;
    error?: string;
  }> {
    try {
      const data = await this.parseContext(query);
      
      return { success: true, data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }
}

// Create singleton instance
export const mlApi = new MLApiService();

export default mlApi;
