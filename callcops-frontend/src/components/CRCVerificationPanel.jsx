/**
 * CRC Verification Panel Component
 * 
 * Displays CRC verification results and attempts error correction.
 */

import { useMemo } from 'react';
import { verifyCRC, findSuspiciousBits, attemptCorrection, messageToHex } from '../utils/crc';

export function CRCVerificationPanel({ decodedMessage, originalMessage = null }) {
  const verification = useMemo(() => {
    if (!decodedMessage || decodedMessage.length !== 128) {
      return null;
    }

    const result = verifyCRC(decodedMessage);
    const suspicious = findSuspiciousBits(decodedMessage, 0.7);
    const correction = attemptCorrection(decodedMessage);

    return {
      ...result,
      suspicious,
      correction,
      messageHex: messageToHex(decodedMessage),
      dataHex: messageToHex(decodedMessage.slice(0, 112))
    };
  }, [decodedMessage]);

  if (!verification) {
    return null;
  }

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <span className="text-lg">🔒</span>
          CRC 검증
        </h3>

        {/* Status Badge */}
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium
          ${verification.isValid
            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
            : verification.correction.success
              ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
              : 'bg-red-500/20 text-red-400 border border-red-500/30'
          }`}
        >
          {verification.isValid
            ? '✓ Valid'
            : verification.correction.success
              ? '⚠ Corrected'
              : '✗ Invalid'
          }
        </span>
      </div>

      {/* CRC Details */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">Expected CRC-16</p>
          <p className="font-mono text-sm text-gray-300">
            0x{verification.expectedCRC.toString(16).padStart(4, '0').toUpperCase()}
          </p>
        </div>
        <div className="bg-surface/50 rounded-lg p-3">
          <p className="text-[10px] text-gray-500 mb-1">Actual CRC-16</p>
          <p className={`font-mono text-sm ${verification.isValid ? 'text-green-400' : 'text-red-400'}`}>
            0x{verification.actualCRC.toString(16).padStart(4, '0').toUpperCase()}
          </p>
        </div>
      </div>

      {/* Confidence */}
      <div>
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>Average Confidence</span>
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

      {/* Error Correction Result */}
      {!verification.isValid && (
        <div className={`rounded-lg p-3 ${verification.correction.success
          ? 'bg-green-500/10 border border-green-500/30'
          : 'bg-red-500/10 border border-red-500/30'
          }`}>
          {verification.correction.success ? (
            <>
              <p className="text-xs text-green-400 mb-1">
                ✓ 단일 비트 오류 정정 성공
              </p>
              <p className="text-[10px] text-gray-400">
                비트 #{verification.correction.correctedBit}를 플립하여 CRC 일치
              </p>
            </>
          ) : (
            <>
              <p className="text-xs text-red-400 mb-1">
                ✗ 오류 정정 실패
              </p>
              <p className="text-[10px] text-gray-400">
                다중 비트 오류 또는 데이터 손상 가능성
              </p>
            </>
          )}
        </div>
      )}

      {/* Data Hex (collapsible) */}
      <details className="cursor-pointer">
        <summary className="text-xs text-gray-500 hover:text-gray-400">
          Payload Hex (112 bits)
        </summary>
        <div className="mt-2 p-2 bg-surface/50 rounded font-mono text-[10px] text-gray-400 break-all">
          {verification.dataHex}
        </div>
      </details>
    </div>
  );
}

export default CRCVerificationPanel;
