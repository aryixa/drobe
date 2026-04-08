/**
 * React hooks for ML API integration
 * Provides easy-to-use hooks for wardrobe recommendations and image analysis
 */

import { useState, useEffect, useCallback } from 'react';
import { mlApi, RecommendationResponse, ImageAnalysisResponse, ContextParseResponse } from '../services/mlApi';

export interface UseRecommendationState {
  recommendation: RecommendationResponse | null;
  loading: boolean;
  error: string | null;
  processingTime: number;
}

export interface UseImageAnalysisState {
  analysis: ImageAnalysisResponse | null;
  loading: boolean;
  error: string | null;
  processingTime: number;
}

export interface UseContextParseState {
  context: ContextParseResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * Hook for getting outfit recommendations
 */
export function useRecommendation() {
  const [state, setState] = useState<UseRecommendationState>({
    recommendation: null,
    loading: false,
    error: null,
    processingTime: 0,
  });

  const getRecommendation = useCallback(async (query: string, maxOutfits: number = 5) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const result = await mlApi.getRecommendationSafe(query, maxOutfits);
      
      if (result.success && result.data) {
        setState({
          recommendation: result.data,
          loading: false,
          error: null,
          processingTime: result.data.processing_time,
        });
      } else {
        setState({
          recommendation: null,
          loading: false,
          error: result.error || 'Failed to get recommendation',
          processingTime: 0,
        });
      }
    } catch (error) {
      setState({
        recommendation: null,
        loading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime: 0,
      });
    }
  }, []);

  const clearRecommendation = useCallback(() => {
    setState({
      recommendation: null,
      loading: false,
      error: null,
      processingTime: 0,
    });
  }, []);

  return {
    ...state,
    getRecommendation,
    clearRecommendation,
  };
}

/**
 * Hook for analyzing clothing images
 */
export function useImageAnalysis() {
  const [state, setState] = useState<UseImageAnalysisState>({
    analysis: null,
    loading: false,
    error: null,
    processingTime: 0,
  });

  const analyzeImage = useCallback(async (
    imagePath: string, 
    analysisType: 'colors' | 'type' | 'pattern' | 'all' = 'all'
  ) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const result = await mlApi.analyzeImageSafe(imagePath, analysisType);
      
      if (result.success && result.data) {
        setState({
          analysis: result.data,
          loading: false,
          error: null,
          processingTime: result.data.processing_time,
        });
      } else {
        setState({
          analysis: null,
          loading: false,
          error: result.error || 'Failed to analyze image',
          processingTime: 0,
        });
      }
    } catch (error) {
      setState({
        analysis: null,
        loading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime: 0,
      });
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    setState({
      analysis: null,
      loading: false,
      error: null,
      processingTime: 0,
    });
  }, []);

  return {
    ...state,
    analyzeImage,
    clearAnalysis,
  };
}

/**
 * Hook for parsing context from natural language
 */
export function useContextParser() {
  const [state, setState] = useState<UseContextParseState>({
    context: null,
    loading: false,
    error: null,
  });

  const parseContext = useCallback(async (query: string) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const result = await mlApi.parseContextSafe(query);
      
      if (result.success && result.data) {
        setState({
          context: result.data,
          loading: false,
          error: null,
        });
      } else {
        setState({
          context: null,
          loading: false,
          error: result.error || 'Failed to parse context',
        });
      }
    } catch (error) {
      setState({
        context: null,
        loading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }, []);

  const clearContext = useCallback(() => {
    setState({
      context: null,
      loading: false,
      error: null,
    });
  }, []);

  return {
    ...state,
    parseContext,
    clearContext,
  };
}

/**
 * Hook for checking API health status
 */
export function useApiHealth() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [capabilities, setCapabilities] = useState<string[]>([]);

  const checkHealth = useCallback(async () => {
    try {
      const health = await mlApi.checkHealth();
      setIsHealthy(health.status === 'healthy');
      setCapabilities(health.capabilities);
      setLastChecked(new Date());
    } catch (error) {
      setIsHealthy(false);
      setCapabilities([]);
      setLastChecked(new Date());
    }
  }, []);

  // Check health on mount
  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Auto-refresh health every 30 seconds
  useEffect(() => {
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return {
    isHealthy,
    lastChecked,
    capabilities,
    checkHealth,
  };
}

/**
 * Hook for batch recommendations
 */
export function useBatchRecommendations() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const [processingTime, setProcessingTime] = useState(0);

  const processBatch = useCallback(async (queries: string[]) => {
    setLoading(true);
    setError(null);

    try {
      const result = await mlApi.batchRecommend({ queries });
      
      if (result.success) {
        setResults(result.results);
        setProcessingTime(result.total_time);
      } else {
        setError(result.errors.join(', ') || 'Batch processing failed');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
    setProcessingTime(0);
  }, []);

  return {
    loading,
    error,
    results,
    processingTime,
    processBatch,
    clearResults,
  };
}

/**
 * Hook for real-time recommendation updates
 */
export function useRealTimeRecommendation(query: string, debounceMs: number = 500) {
  const [state, setState] = useState<UseRecommendationState>({
    recommendation: null,
    loading: false,
    error: null,
    processingTime: 0,
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      if (query.trim()) {
        setState(prev => ({ ...prev, loading: true, error: null }));
        
        mlApi.getRecommendationSafe(query, 3).then(result => {
          if (result.success && result.data) {
            setState({
              recommendation: result.data,
              loading: false,
              error: null,
              processingTime: result.data.processing_time,
            });
          } else {
            setState({
              recommendation: null,
              loading: false,
              error: result.error || 'Failed to get recommendation',
              processingTime: 0,
            });
          }
        }).catch(error => {
          setState({
            recommendation: null,
            loading: false,
            error: error instanceof Error ? error.message : 'Unknown error',
            processingTime: 0,
          });
        });
      } else {
        setState({
          recommendation: null,
          loading: false,
          error: null,
          processingTime: 0,
        });
      }
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [query, debounceMs]);

  return state;
}

/**
 * Hook for managing recommendation history
 */
export function useRecommendationHistory() {
  const [history, setHistory] = useState<RecommendationResponse[]>([]);

  const addToHistory = useCallback((recommendation: RecommendationResponse) => {
    setHistory(prev => [recommendation, ...prev.slice(0, 9)]); // Keep last 10
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  const removeFromHistory = useCallback((index: number) => {
    setHistory(prev => prev.filter((_, i) => i !== index));
  }, []);

  return {
    history,
    addToHistory,
    clearHistory,
    removeFromHistory,
  };
}
