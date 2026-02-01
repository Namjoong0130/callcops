/**
 * Audio Comparison Panel Component
 * 
 * Displays similarity metrics between original and watermarked audio.
 */

import { useMemo, useCallback } from 'react';
import { calculateAllMetrics, getQualityRating } from '../utils/audioComparison';

export function AudioComparisonPanel({ originalAudio, watermarkedAudio }) {
  const metrics = useMemo(() => {
    if (!originalAudio || !watermarkedAudio) {
      return null;
    }
    return calculateAllMetrics(originalAudio, watermarkedAudio);
  }, [originalAudio, watermarkedAudio]);
  
  const qualityRating = useMemo(() => {
    if (!metrics) return null;
    return getQualityRating(metrics.snr);
  }, [metrics]);
  
  if (!metrics) {
    return null;
  }
  
  const formatValue = (value, decimals = 2) => {
    if (value === Infinity) return '∞';
    if (value === -Infinity) return '-∞';
    return value.toFixed(decimals);
  };
  
  const getColorForSNR = (snr) => {
    if (snr >= 40) return 'text-green-400';
    if (snr >= 30) return 'text-lime-400';
    if (snr >= 20) return 'text-yellow-400';
    if (snr >= 10) return 'text-orange-400';
    return 'text-red-400';
  };
  
  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <span className="text-lg">📊</span>
          원본 vs 워터마크 비교
        </h3>
        
        {/* Quality Badge */}
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium
          bg-${qualityRating.color}-500/20 text-${qualityRating.color}-400 
          border border-${qualityRating.color}-500/30`}
          style={{
            backgroundColor: `rgb(var(--color-${qualityRating.color}-500) / 0.2)`,
            color: `rgb(var(--color-${qualityRating.color}-400))`,
          }}
        >
          {qualityRating.rating}
        </span>
      </div>
      
      {/* Main Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        {/* SNR */}
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">Signal-to-Noise Ratio</p>
          <p className={`text-2xl font-mono font-bold ${getColorForSNR(metrics.snr)}`}>
            {formatValue(metrics.snr, 1)}
            <span className="text-sm text-gray-500 ml-1">dB</span>
          </p>
          <p className="text-[10px] text-gray-500 mt-1">
            {qualityRating.description}
          </p>
        </div>
        
        {/* PSNR */}
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">Peak SNR</p>
          <p className={`text-2xl font-mono font-bold ${getColorForSNR(metrics.psnr - 10)}`}>
            {formatValue(metrics.psnr, 1)}
            <span className="text-sm text-gray-500 ml-1">dB</span>
          </p>
          <p className="text-[10px] text-gray-500 mt-1">
            Higher = Better
          </p>
        </div>
        
        {/* Correlation */}
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">Correlation</p>
          <p className={`text-2xl font-mono font-bold ${
            metrics.correlation > 0.999 ? 'text-green-400' :
            metrics.correlation > 0.99 ? 'text-lime-400' :
            metrics.correlation > 0.95 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {(metrics.correlation * 100).toFixed(3)}
            <span className="text-sm text-gray-500 ml-1">%</span>
          </p>
          <p className="text-[10px] text-gray-500 mt-1">
            Waveform Similarity
          </p>
        </div>
        
        {/* RMSE */}
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">RMSE</p>
          <p className={`text-2xl font-mono font-bold ${
            metrics.rmse < 0.001 ? 'text-green-400' :
            metrics.rmse < 0.01 ? 'text-lime-400' :
            metrics.rmse < 0.05 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {formatValue(metrics.rmse, 4)}
          </p>
          <p className="text-[10px] text-gray-500 mt-1">
            Lower = Better
          </p>
        </div>
      </div>
      
      {/* Visual SNR Bar */}
      <div>
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>음질 등급</span>
          <span>SNR: {formatValue(metrics.snr)}dB</span>
        </div>
        <div className="h-3 bg-surface rounded-full overflow-hidden relative">
          {/* Background gradient showing scale */}
          <div className="absolute inset-0 flex">
            <div className="flex-1 bg-red-500/30" title="Bad (<10dB)" />
            <div className="flex-1 bg-orange-500/30" title="Poor (10-20dB)" />
            <div className="flex-1 bg-yellow-500/30" title="Fair (20-30dB)" />
            <div className="flex-1 bg-lime-500/30" title="Good (30-40dB)" />
            <div className="flex-1 bg-green-500/30" title="Excellent (40+dB)" />
          </div>
          {/* SNR indicator */}
          <div 
            className="absolute top-0 bottom-0 w-1 bg-white shadow-lg transition-all duration-300"
            style={{ 
              left: `${Math.min(100, Math.max(0, (metrics.snr / 50) * 100))}%`,
            }}
          />
        </div>
        <div className="flex justify-between text-[9px] text-gray-600 mt-1">
          <span>0dB</span>
          <span>10dB</span>
          <span>20dB</span>
          <span>30dB</span>
          <span>40dB</span>
          <span>50dB</span>
        </div>
      </div>
      
      {/* Resampling Info */}
      <div className="bg-surface/30 rounded-lg p-3">
        <p className="text-[10px] text-gray-500 mb-1">리샘플링 정보</p>
        <p className="text-xs text-gray-400">
          <span className="text-primary">Web Audio API</span>를 사용하여 8kHz로 리샘플링
        </p>
        <p className="text-[10px] text-gray-500 mt-1">
          OfflineAudioContext 기반 고품질 리샘플링 (Polyphase Interpolation)
        </p>
      </div>
      
      {/* Duration Info */}
      <div className="flex gap-3 text-[10px] text-gray-500">
        <span>원본: {metrics.durationOriginal.toFixed(2)}초</span>
        <span>|</span>
        <span>워터마크: {metrics.durationWatermarked.toFixed(2)}초</span>
        <span>|</span>
        <span>샘플: {originalAudio.length.toLocaleString()}</span>
      </div>
    </div>
  );
}

export default AudioComparisonPanel;
