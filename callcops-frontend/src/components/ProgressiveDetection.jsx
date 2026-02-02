/**
 * Progressive Detection Component
 *
 * Pure display component driven by parent-computed detectionState.
 * Shows accumulated 128-bit confidence grid synced with audio playback.
 * - Cycle progress bar (5.12s per cycle)
 * - Overall progress bar
 * - 128-bit grid with grayscale confidence
 */

const PAYLOAD_LENGTH = 128;
const SAMPLE_RATE = 8000;
const FRAME_SAMPLES = 320;  // 40ms @ 8kHz

export function ProgressiveDetection({
  audioData,
  isDetecting,
  detectionState,
  currentFrameIndex,
  totalFrames,
  onStartDetection,
  onStopDetection,
  isModelReady,
}) {
  // Audio duration
  const audioDuration = audioData ? audioData.length / SAMPLE_RATE : 0;

  // Progress percentage
  const progress = totalFrames > 0 && currentFrameIndex >= 0
    ? Math.min(100, ((currentFrameIndex + 1) / totalFrames) * 100)
    : 0;

  // Cycle info
  const cycleFrame = detectionState?.cycleFrame ?? -1;
  const cycleCount = detectionState?.cycleCount ?? 0;
  const cycleProgress = cycleFrame >= 0 ? ((cycleFrame + 1) / PAYLOAD_LENGTH) * 100 : 0;

  // Get accumulated bit probs
  const accumulated = detectionState?.accumulated;
  const counts = detectionState?.counts;

  // Bit style
  const getBitStyle = (prob, hasData) => {
    if (!hasData || prob === undefined) {
      return { backgroundColor: '#374151', color: '#9CA3AF' };
    }
    const lightness = Math.floor(prob * 100);
    const backgroundColor = `hsl(0, 0%, ${lightness}%)`;
    const color = prob < 0.5 ? '#FFFFFF' : '#000000';
    return { backgroundColor, color };
  };

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          {isDetecting && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
          Progressive Detection
        </h3>
        {isDetecting && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              Cycle {cycleCount + 1}
            </span>
            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded font-mono">
              {cycleFrame + 1}/128
            </span>
          </div>
        )}
      </div>

      {/* Overall Progress Bar */}
      <div>
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>Frame {currentFrameIndex >= 0 ? currentFrameIndex + 1 : 0} / {totalFrames}</span>
          <span>{progress.toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-surface rounded-full overflow-hidden relative">
          <div
            className="h-full bg-gradient-to-r from-gray-600 to-gray-200 transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
          {isDetecting && (
            <div
              className="absolute top-0 bottom-0 w-1 bg-white shadow-lg animate-pulse"
              style={{ left: `${progress}%` }}
            />
          )}
        </div>
      </div>

      {/* Cycle Progress (5.12s cycle) */}
      {isDetecting && (
        <div>
          <div className="flex justify-between text-[10px] text-gray-500 mb-1">
            <span>Cycle Progress</span>
            <span>{cycleProgress.toFixed(0)}% ({((cycleFrame + 1) * 40 / 1000).toFixed(2)}s / 5.12s)</span>
          </div>
          <div className="h-1.5 bg-surface rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-cyan-400 transition-all duration-100"
              style={{ width: `${cycleProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* 128-bit Accumulated Grid */}
      <div className="grid grid-cols-16 gap-1">
        {Array.from({ length: 128 }).map((_, i) => {
          const prob = accumulated ? accumulated[i] : 0.5;
          const hasData = counts ? counts[i] > 0 : false;
          const isCurrent = isDetecting && (currentFrameIndex % PAYLOAD_LENGTH) === i;
          const style = getBitStyle(prob, hasData);

          return (
            <div
              key={i}
              className={`aspect-square rounded-sm flex items-center justify-center text-[11px] font-mono font-bold
                transition-all duration-200
                ${!hasData ? 'opacity-30' : ''}
                ${isCurrent ? 'ring-1 ring-white animate-pulse' : ''}`}
              style={style}
            >
              {hasData ? (prob > 0.5 ? '1' : '0') : '?'}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 text-[10px] text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm border border-gray-600" style={{ backgroundColor: '#ffffff' }} />
          1 (White)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#808080' }} />
          Uncertain (Gray)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm border border-gray-600" style={{ backgroundColor: '#000000' }} />
          0 (Black)
        </span>
      </div>

      {/* Control Button */}
      {!isDetecting ? (
        <button
          onClick={onStartDetection}
          disabled={!audioData || !isModelReady}
          className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all
            ${audioData && isModelReady
              ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:opacity-90 glow'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
          실시간 탐지 시작
        </button>
      ) : (
        <button
          onClick={onStopDetection}
          className="w-full py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400
                   rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
          탐지 중지
        </button>
      )}

      {/* Stats */}
      {!isDetecting && totalFrames > 0 && currentFrameIndex >= 0 && (
        <div className="flex justify-center gap-4 text-[10px] text-gray-500">
          <span>{totalFrames} 프레임</span>
          <span>|</span>
          <span>{audioDuration.toFixed(2)}초 음원</span>
          <span>|</span>
          <span>{cycleCount + 1} 사이클</span>
        </div>
      )}
    </div>
  );
}

export default ProgressiveDetection;
