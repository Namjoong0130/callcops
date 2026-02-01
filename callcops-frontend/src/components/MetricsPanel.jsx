/**
 * MetricsPanel Component
 * 
 * Displays detection metrics: confidence, status, and stats.
 */

export function MetricsPanel({
  bitProbs,
  status = 'idle',
  inferenceTime = 0,
  error = null
}) {
  /**
   * Calculate confidence score
   * Confidence = average of max(p, 1-p) for all 128 bits
   */
  const calculateConfidence = () => {
    if (!bitProbs || bitProbs.length !== 128) return 0;

    let sum = 0;
    for (let i = 0; i < 128; i++) {
      sum += Math.max(bitProbs[i], 1 - bitProbs[i]);
    }
    return (sum / 128) * 100;
  };

  /**
   * Calculate detection score
   * Higher when bits are clearly 0 or 1, lower when uncertain
   */
  const calculateDetection = () => {
    if (!bitProbs || bitProbs.length !== 128) return 0;

    let sum = 0;
    for (let i = 0; i < 128; i++) {
      sum += Math.abs(bitProbs[i] - 0.5) * 2;
    }
    return (sum / 128) * 100;
  };

  /**
   * Count bits that are confidently detected
   */
  const countConfidentBits = () => {
    if (!bitProbs) return { zeros: 0, ones: 0, uncertain: 128 };

    let zeros = 0;
    let ones = 0;
    let uncertain = 0;

    for (let i = 0; i < 128; i++) {
      if (bitProbs[i] > 0.7) ones++;
      else if (bitProbs[i] < 0.3) zeros++;
      else uncertain++;
    }

    return { zeros, ones, uncertain };
  };

  const confidence = calculateConfidence();
  const detection = calculateDetection();
  const bitCounts = countConfidentBits();

  const getStatusColor = () => {
    switch (status) {
      case 'recording': return 'text-red-400';
      case 'processing': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'recording':
        return (
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
          </span>
        );
      case 'processing':
        return (
          <svg className="animate-spin h-4 w-4 text-yellow-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        );
      default:
        return (
          <span className="inline-flex rounded-full h-3 w-3 bg-gray-500"></span>
        );
    }
  };

  return (
    <div className="glass rounded-xl p-4 space-y-4 h-full">
      <h3 className="text-sm font-semibold text-gray-300">Detection Metrics</h3>

      {/* Status */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">Status</span>
        <div className={`flex items-center gap-2 ${getStatusColor()}`}>
          {getStatusIcon()}
          <span className="text-sm font-medium capitalize">{status}</span>
        </div>
      </div>

      {/* Confidence */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Confidence</span>
          <span className="text-lg font-bold text-primary">
            {confidence.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-gradient-to-r from-primary to-purple-400 h-2 rounded-full transition-all duration-300"
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      {/* Detection */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Detection</span>
          <span className="text-lg font-bold text-green-400">
            {detection.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-gradient-to-r from-green-500 to-emerald-400 h-2 rounded-full transition-all duration-300"
            style={{ width: `${detection}%` }}
          />
        </div>
      </div>

      {/* Bit Breakdown */}
      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-gray-700">
        <div className="text-center">
          <div className="text-lg font-bold text-red-400">{bitCounts.zeros}</div>
          <div className="text-xs text-gray-500">Zeros</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-yellow-400">{bitCounts.uncertain}</div>
          <div className="text-xs text-gray-500">Uncertain</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-400">{bitCounts.ones}</div>
          <div className="text-xs text-gray-500">Ones</div>
        </div>
      </div>

      {/* Inference Time */}
      {inferenceTime > 0 && (
        <div className="flex items-center justify-between pt-2 border-t border-gray-700">
          <span className="text-xs text-gray-500">Inference Time</span>
          <span className="text-xs font-mono text-gray-400">
            {inferenceTime.toFixed(1)}ms
          </span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-2 bg-red-900/30 border border-red-500/30 rounded-lg">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
}

export default MetricsPanel;
