/**
 * RS Verification Panel Component
 * 
 * Displays Reed-Solomon verification results with 2-byte error correction.
 * Replaces CRCVerificationPanel with proper RS(16,12) error correction.
 */

import { useMemo } from 'react';
import { verifyRS, findSuspiciousBits, messageToHex, extractDataFields } from '../utils/reedSolomon';

export function RSVerificationPanel({ decodedMessage, originalMessage = null }) {
  const verification = useMemo(() => {
    if (!decodedMessage || decodedMessage.length !== 128) {
      return null;
    }

    const result = verifyRS(decodedMessage);
    const suspicious = findSuspiciousBits(decodedMessage, 0.7);
    const fields = extractDataFields(result.corrected);

    return {
      isValid: result.isValid,
      errorsCorrected: result.errorsCorrected,
      confidence: result.confidence,
      corrected: result.corrected,
      suspicious,
      messageHex: messageToHex(result.corrected),
      dataHex: messageToHex(result.corrected.slice(0, 96)),
      fields
    };
  }, [decodedMessage]);

  if (!verification) {
    return null;
  }

  // Determine status
  const wasAutoFixed = verification.isValid && verification.errorsCorrected > 0;
  const isFailed = !verification.isValid;

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <span className="text-lg">🛡️</span>
          Reed-Solomon 검증
        </h3>

        {/* Status Badge */}
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium
          ${!isFailed && !wasAutoFixed
            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
            : wasAutoFixed
              ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
              : 'bg-red-500/20 text-red-400 border border-red-500/30'
          }`}
        >
          {!isFailed && !wasAutoFixed
            ? '✓ Valid'
            : wasAutoFixed
              ? `⚠ Fixed (${verification.errorsCorrected} byte)`
              : '✗ Uncorrectable'
          }
        </span>
      </div>

      {/* RS Info */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">RS 코드</p>
          <p className="font-mono text-sm text-gray-300">
            RS(16,12) / GF(2⁸)
          </p>
        </div>
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">정정 능력</p>
          <p className={`font-mono text-sm ${verification.isValid ? 'text-green-400' : 'text-red-400'}`}>
            최대 2 bytes (16 bits)
          </p>
        </div>
      </div>

      {/* Error Correction Status */}
      {verification.errorsCorrected > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-xs text-yellow-400 mb-1">
            ✓ 오류 정정 완료
          </p>
          <p className="text-[10px] text-gray-400">
            {verification.errorsCorrected}개 바이트 오류를 자동 정정했습니다
          </p>
        </div>
      )}

      {isFailed && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
          <p className="text-xs text-red-400 mb-1">
            ✗ 오류 정정 실패
          </p>
          <p className="text-[10px] text-gray-400">
            3바이트 이상의 오류로 정정 불가능
          </p>
        </div>
      )}

      {/* Confidence */}
      <div>
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>평균 신뢰도</span>
          <span>{(verification.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2 bg-surface rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${verification.confidence > 0.8 ? 'bg-green-500' :
              verification.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
            style={{ width: `${verification.confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Suspicious Bits */}
      {verification.suspicious.length > 0 && (
        <div className="bg-surface/30 rounded-lg p-3">
          <p className="text-xs text-yellow-400 mb-2">
            ⚠ 신뢰도가 낮은 비트: {verification.suspicious.length}개
          </p>
          <div className="flex flex-wrap gap-1">
            {verification.suspicious.slice(0, 10).map(({ index, confidence }) => (
              <span
                key={index}
                className="px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded text-[10px] font-mono"
                title={`Confidence: ${(confidence * 100).toFixed(1)}%`}
              >
                #{index}
              </span>
            ))}
            {verification.suspicious.length > 10 && (
              <span className="text-[10px] text-gray-500">
                +{verification.suspicious.length - 10} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Message Structure */}
      <details className="cursor-pointer">
        <summary className="text-xs text-gray-500 hover:text-gray-400">
          메시지 구조 (96 bits data + 32 bits RS)
        </summary>
        <div className="mt-2 space-y-2">
          <div className="p-2 bg-surface/50 rounded">
            <p className="text-[10px] text-gray-500 mb-1">Sync (16 bits)</p>
            <p className="font-mono text-[10px] text-green-400">
              {verification.fields.sync.join('')}
            </p>
          </div>
          <div className="p-2 bg-surface/50 rounded">
            <p className="text-[10px] text-gray-500 mb-1">Timestamp (32 bits)</p>
            <p className="font-mono text-[10px] text-blue-400">
              {verification.fields.timestamp.join('')}
            </p>
          </div>
          <div className="p-2 bg-surface/50 rounded">
            <p className="text-[10px] text-gray-500 mb-1">Auth (48 bits)</p>
            <p className="font-mono text-[10px] text-purple-400 break-all">
              {verification.fields.auth.join('')}
            </p>
          </div>
          <div className="p-2 bg-surface/50 rounded">
            <p className="text-[10px] text-gray-500 mb-1">Full Hex</p>
            <p className="font-mono text-[10px] text-gray-400 break-all">
              {verification.messageHex}
            </p>
          </div>
        </div>
      </details>
    </div>
  );
}

// Also export as CRCVerificationPanel for backward compatibility
export const CRCVerificationPanel = RSVerificationPanel;

export default RSVerificationPanel;
