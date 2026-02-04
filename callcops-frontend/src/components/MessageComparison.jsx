/**
 * MessageComparison Component
 * 
 * Visualizes the comparison between original embedded message
 * and decoded message from the detector.
 */

import { useMemo } from 'react';

// 128-bit message structure (RS)
const SYNC_BITS = 16;
const TIMESTAMP_BITS = 32;
const AUTH_BITS = 48;  // Reduced from 64 for RS parity
const RS_BITS = 32;    // Reed-Solomon parity (was CRC 16)

export function MessageComparison({
  originalMessage,  // Float32Array[128] - embedded message (0 or 1)
  decodedProbs,     // Float32Array[128] - decoded probabilities (0~1)
  correctedBitIndex = null, // number|null - index of CRC-corrected bit
  showDetails = true
}) {
  // Calculate metrics
  const metrics = useMemo(() => {
    if (!originalMessage || !decodedProbs || originalMessage.length !== 128 || decodedProbs.length !== 128) {
      return null;
    }

    let matches = 0;
    let syncMatches = 0;
    let timestampMatches = 0;
    let authMatches = 0;
    let rsMatches = 0;

    const comparison = [];

    for (let i = 0; i < 128; i++) {
      const original = originalMessage[i] > 0.5 ? 1 : 0;
      const decoded = decodedProbs[i] > 0.5 ? 1 : 0;
      const isMatch = original === decoded;
      const confidence = Math.abs(decodedProbs[i] - 0.5) * 2;

      comparison.push({
        index: i,
        original,
        decoded,
        prob: decodedProbs[i],
        isMatch,
        confidence
      });

      if (isMatch) {
        matches++;
        if (i < SYNC_BITS) syncMatches++;
        else if (i < SYNC_BITS + TIMESTAMP_BITS) timestampMatches++;
        else if (i < SYNC_BITS + TIMESTAMP_BITS + AUTH_BITS) authMatches++;
        else rsMatches++;
      }
    }

    return {
      totalAccuracy: (matches / 128 * 100).toFixed(1),
      syncAccuracy: (syncMatches / SYNC_BITS * 100).toFixed(1),
      timestampAccuracy: (timestampMatches / TIMESTAMP_BITS * 100).toFixed(1),
      authAccuracy: (authMatches / AUTH_BITS * 100).toFixed(1),
      rsAccuracy: (rsMatches / RS_BITS * 100).toFixed(1),
      matches,
      comparison
    };
  }, [originalMessage, decodedProbs]);

  if (!metrics) {
    return (
      <div className="text-center text-gray-500 py-8">
        <p>비교할 데이터가 없습니다</p>
        <p className="text-xs mt-1">먼저 Embed 후 Verify를 실행해주세요</p>
      </div>
    );
  }

  const getAccuracyColor = (acc) => {
    const num = parseFloat(acc);
    if (num >= 95) return 'text-green-400';
    if (num >= 80) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getAccuracyBg = (acc) => {
    const num = parseFloat(acc);
    if (num >= 95) return 'bg-green-500';
    if (num >= 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-4">
      {/* Overall Accuracy */}
      <div className="text-center">
        <p className="text-xs text-gray-400 mb-1">전체 일치율</p>
        <p className={`text-4xl font-bold ${getAccuracyColor(metrics.totalAccuracy)}`}>
          {metrics.totalAccuracy}%
        </p>
        <p className="text-xs text-gray-500 mt-1">
          {metrics.matches} / 128 비트 일치
        </p>
      </div>

      {/* Section Breakdown */}
      <div className="grid grid-cols-4 gap-2 text-center">
        <div className="bg-surface/50 rounded-lg p-2">
          <p className="text-[10px] text-gray-500 mb-1">Sync</p>
          <p className={`text-lg font-bold ${getAccuracyColor(metrics.syncAccuracy)}`}>
            {metrics.syncAccuracy}%
          </p>
          <div className="h-1 bg-gray-700 rounded-full mt-1 overflow-hidden">
            <div
              className={`h-full ${getAccuracyBg(metrics.syncAccuracy)} transition-all`}
              style={{ width: `${metrics.syncAccuracy}%` }}
            />
          </div>
        </div>

        <div className="bg-surface/50 rounded-lg p-2">
          <p className="text-[10px] text-gray-500 mb-1">Timestamp</p>
          <p className={`text-lg font-bold ${getAccuracyColor(metrics.timestampAccuracy)}`}>
            {metrics.timestampAccuracy}%
          </p>
          <div className="h-1 bg-gray-700 rounded-full mt-1 overflow-hidden">
            <div
              className={`h-full ${getAccuracyBg(metrics.timestampAccuracy)} transition-all`}
              style={{ width: `${metrics.timestampAccuracy}%` }}
            />
          </div>
        </div>

        <div className="bg-surface/50 rounded-lg p-2">
          <p className="text-[10px] text-gray-500 mb-1">Auth</p>
          <p className={`text-lg font-bold ${getAccuracyColor(metrics.authAccuracy)}`}>
            {metrics.authAccuracy}%
          </p>
          <div className="h-1 bg-gray-700 rounded-full mt-1 overflow-hidden">
            <div
              className={`h-full ${getAccuracyBg(metrics.authAccuracy)} transition-all`}
              style={{ width: `${metrics.authAccuracy}%` }}
            />
          </div>
        </div>

        <div className="bg-surface/50 rounded-lg p-2">
          <p className="text-[10px] text-gray-500 mb-1">RS Parity</p>
          <p className={`text-lg font-bold ${getAccuracyColor(metrics.rsAccuracy)}`}>
            {metrics.rsAccuracy}%
          </p>
          <div className="h-1 bg-gray-700 rounded-full mt-1 overflow-hidden">
            <div
              className={`h-full ${getAccuracyBg(metrics.rsAccuracy)} transition-all`}
              style={{ width: `${metrics.rsAccuracy}%` }}
            />
          </div>
        </div>
      </div>

      {/* Bit-by-bit comparison grid */}
      {showDetails && (
        <div>
          <p className="text-xs text-gray-400 mb-2">
            비트별 비교 (녹색: 일치, 빨강: 불일치{correctedBitIndex !== null ? ', 노랑: RS 정정' : ''})
          </p>
          <div className="grid grid-cols-16 gap-0.5">
            {metrics.comparison.map((bit, i) => {
              const isCorrected = correctedBitIndex === i;
              const correctedValue = isCorrected ? (bit.decoded === 1 ? 0 : 1) : null;

              return (
                <div
                  key={i}
                  className={`w-full aspect-square rounded-sm flex items-center justify-center text-[8px] font-mono relative
                    ${isCorrected
                      ? 'bg-amber-500/40 text-amber-100 ring-2 ring-amber-400/60'
                      : bit.isMatch
                        ? 'bg-green-500/30 text-green-300'
                        : 'bg-red-500/50 text-red-200'
                    }
                    ${!isCorrected && i < SYNC_BITS ? 'ring-1 ring-cyan-500/30' : ''}
                    ${!isCorrected && i >= SYNC_BITS && i < SYNC_BITS + TIMESTAMP_BITS ? 'ring-1 ring-yellow-500/30' : ''}
                    ${!isCorrected && i >= SYNC_BITS + TIMESTAMP_BITS && i < SYNC_BITS + TIMESTAMP_BITS + AUTH_BITS ? 'ring-1 ring-green-500/30' : ''}
                    ${!isCorrected && i >= SYNC_BITS + TIMESTAMP_BITS + AUTH_BITS ? 'ring-1 ring-purple-500/30' : ''}
                  `}
                  title={isCorrected
                    ? `Bit ${i}: RS 오류 정정됨 (${bit.decoded} → ${correctedValue})`
                    : `Bit ${i}: Original=${bit.original}, Decoded=${bit.decoded} (${(bit.prob * 100).toFixed(0)}%)`
                  }
                >
                  {isCorrected ? correctedValue : bit.decoded}
                  {isCorrected && (
                    <span className="absolute -top-0.5 -right-0.5 text-[6px] text-amber-300 leading-none">
                      ✓
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="flex justify-center gap-4 mt-2 text-[10px] text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-green-500/30" /> Match
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-red-500/50" /> Mismatch
            </span>
            {correctedBitIndex !== null && (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-amber-500/40 ring-1 ring-amber-400/60" /> Corrected
              </span>
            )}
          </div>
          <div className="flex justify-center gap-4 mt-1 text-[10px] text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-cyan-500/50" /> Sync
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-yellow-500/50" /> Timestamp
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-green-500/50" /> Auth
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-purple-500/50" /> RS Parity
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default MessageComparison;
