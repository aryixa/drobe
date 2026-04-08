/**
 * Demo component for ML API integration
 * Shows how to use the wardrobe recommendation system
 */

import React, { useState } from 'react';
import { useRecommendation, useApiHealth, useImageAnalysis, useContextParser } from '../hooks/useMLApi';
import { RecommendationResponse } from '../services/mlApi';

export function MLRecommendationDemo() {
  const [query, setQuery] = useState('summer casual day outfit');
  const [imagePath, setImagePath] = useState('');
  const [activeTab, setActiveTab] = useState<'recommendation' | 'analysis' | 'context'>('recommendation');
  
  const {
    recommendation,
    loading: recLoading,
    error: recError,
    getRecommendation,
    clearRecommendation,
  } = useRecommendation();
  
  const {
    analysis,
    loading: analysisLoading,
    error: analysisError,
    analyzeImage,
  } = useImageAnalysis();
  
  const {
    context,
    loading: contextLoading,
    error: contextError,
    parseContext,
  } = useContextParser();
  
  const { isHealthy, capabilities, lastChecked } = useApiHealth();

  const handleGetRecommendation = () => {
    if (query.trim()) {
      getRecommendation(query, 5);
    }
  };

  const handleAnalyzeImage = () => {
    if (imagePath.trim()) {
      analyzeImage(imagePath, 'all');
    }
  };

  const handleParseContext = () => {
    if (query.trim()) {
      parseContext(query);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ML Recommendation Demo</h1>
        <div className="flex items-center space-x-4 text-sm">
          <div className={`px-3 py-1 rounded-full ${
            isHealthy ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            API: {isHealthy ? 'Healthy' : 'Unhealthy'}
          </div>
          {lastChecked && (
            <span className="text-gray-500">
              Last checked: {lastChecked.toLocaleTimeString()}
            </span>
          )}
        </div>
        {capabilities.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {capabilities.map(cap => (
              <span key={cap} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                {cap}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {(['recommendation', 'analysis', 'context'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
      </div>

      {/* Recommendation Tab */}
      {activeTab === 'recommendation' && (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter your outfit request:
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleGetRecommendation()}
                placeholder="e.g., 'summer casual day outfit' or 'formal business meeting attire'"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleGetRecommendation}
                disabled={recLoading || !query.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {recLoading ? 'Processing...' : 'Get Recommendation'}
              </button>
              <button
                onClick={clearRecommendation}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Clear
              </button>
            </div>
          </div>

          {recError && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800">{recError}</p>
            </div>
          )}

          {recommendation && (
            <div className="space-y-4">
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-semibold text-green-800 mb-2">Recommendation Found!</h3>
                <p className="text-sm text-gray-600 mb-2">Query: {recommendation.query}</p>
                <p className="text-sm text-gray-600">Processing time: {recommendation.processing_time.toFixed(2)}s</p>
              </div>

              {recommendation.recommendation && (
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-3">Primary Recommendation</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Overall Score:</span>
                      <span className="text-sm">{recommendation.recommendation.score.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Style Score:</span>
                      <span className="text-sm">{recommendation.recommendation.style_score.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Color Score:</span>
                      <span className="text-sm">{recommendation.recommendation.color_score.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Pattern Score:</span>
                      <span className="text-sm">{recommendation.recommendation.pattern_score.toFixed(3)}</span>
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <h5 className="text-sm font-medium mb-2">Recommended Items:</h5>
                    <ul className="list-disc list-inside text-sm text-gray-600">
                      {recommendation.recommendation.items.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {recommendation.explanation && (
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-3">Explanation</h4>
                  {recommendation.explanation.primary_reason && (
                    <p className="text-sm text-gray-700 mb-2">{recommendation.explanation.primary_reason}</p>
                  )}
                  {recommendation.explanation.supporting_rules && (
                    <div className="mt-2">
                      <h5 className="text-sm font-medium mb-1">Supporting Rules:</h5>
                      <ul className="list-disc list-inside text-xs text-gray-600">
                        {recommendation.explanation.supporting_rules.slice(0, 3).map((rule: any, index: number) => (
                          <li key={index}>{rule.title}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {recommendation.alternatives.length > 0 && (
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-3">Alternatives</h4>
                  <div className="space-y-2">
                    {recommendation.alternatives.map((alt, index) => (
                      <div key={index} className="p-2 bg-gray-50 rounded">
                        <div className="flex justify-between">
                          <span className="text-sm font-medium">Score:</span>
                          <span className="text-sm">{alt.score.toFixed(3)}</span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">
                          Items: {alt.items.join(', ')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Image Analysis Tab */}
      {activeTab === 'analysis' && (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter image path for analysis:
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={imagePath}
                onChange={(e) => setImagePath(e.target.value)}
                placeholder="e.g., /path/to/clothing/image.jpg"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleAnalyzeImage}
                disabled={analysisLoading || !imagePath.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {analysisLoading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </div>
          </div>

          {analysisError && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800">{analysisError}</p>
            </div>
          )}

          {analysis && (
            <div className="space-y-4">
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-semibold text-green-800 mb-2">Analysis Complete!</h3>
                <p className="text-sm text-gray-600">Processing time: {analysis.processing_time.toFixed(2)}s</p>
              </div>

              {analysis.colors && (
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-3">Color Analysis</h4>
                  <div className="space-y-2">
                    {analysis.colors.map((color, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <div 
                          className="w-8 h-8 rounded border border-gray-300"
                          style={{ backgroundColor: `rgb(${color.rgb.join(', ')})` }}
                        />
                        <div>
                          <span className="text-sm font-medium capitalize">{color.category}</span>
                          <div className="text-xs text-gray-500">
                            {(color.percentage * 100).toFixed(1)}% - Confidence: {color.confidence.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {analysis.type_classification && (
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-3">Type Classification</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Primary Type:</span>
                      <span className="text-sm capitalize">{analysis.type_classification.primary_type}</span>
                    </div>
                    {analysis.type_classification.sub_type && (
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">Sub Type:</span>
                        <span className="text-sm capitalize">{analysis.type_classification.sub_type}</span>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Confidence:</span>
                      <span className="text-sm">{analysis.type_classification.confidence.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}

              {analysis.pattern_detection && (
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-3">Pattern Detection</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Pattern Type:</span>
                      <span className="text-sm capitalize">{analysis.pattern_detection.pattern_type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Confidence:</span>
                      <span className="text-sm">{analysis.pattern_detection.confidence.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Context Parser Tab */}
      {activeTab === 'context' && (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter query to parse context:
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., 'formal business meeting for winter'"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleParseContext}
                disabled={contextLoading || !query.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {contextLoading ? 'Parsing...' : 'Parse Context'}
              </button>
            </div>
          </div>

          {contextError && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800">{contextError}</p>
            </div>
          )}

          {context && (
            <div className="space-y-4">
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-semibold text-green-800 mb-2">Context Parsed!</h3>
                <p className="text-sm text-gray-600">Confidence: {(context.confidence * 100).toFixed(1)}%</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-2">Basic Context</h4>
                  <div className="space-y-1 text-sm">
                    {context.occasion && (
                      <div><span className="font-medium">Occasion:</span> {context.occasion}</div>
                    )}
                    {context.season && (
                      <div><span className="font-medium">Season:</span> {context.season}</div>
                    )}
                    {context.weather && (
                      <div><span className="font-medium">Weather:</span> {context.weather}</div>
                    )}
                    {context.time_of_day && (
                      <div><span className="font-medium">Time:</span> {context.time_of_day}</div>
                    )}
                    {context.style_level && (
                      <div><span className="font-medium">Style:</span> {context.style_level}</div>
                    )}
                  </div>
                </div>

                <div className="p-4 border border-gray-200 rounded-lg">
                  <h4 className="font-semibold mb-2">Extracted Elements</h4>
                  <div className="space-y-1 text-sm">
                    {context.colors.length > 0 && (
                      <div>
                        <span className="font-medium">Colors:</span> {context.colors.join(', ')}
                      </div>
                    )}
                    {context.patterns.length > 0 && (
                      <div>
                        <span className="font-medium">Patterns:</span> {context.patterns.join(', ')}
                      </div>
                    )}
                    {context.clothing_types.length > 0 && (
                      <div>
                        <span className="font-medium">Types:</span> {context.clothing_types.join(', ')}
                      </div>
                    )}
                    {context.keywords.length > 0 && (
                      <div>
                        <span className="font-medium">Keywords:</span> {context.keywords.join(', ')}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {context.parsing_errors.length > 0 && (
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h4 className="font-semibold text-yellow-800 mb-2">Parsing Notes</h4>
                  <ul className="list-disc list-inside text-sm text-yellow-700">
                    {context.parsing_errors.map((error, index) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
