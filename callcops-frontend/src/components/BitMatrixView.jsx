/**
 * BitMatrixView Component
 * 
 * Displays 8x16 grid representing 128 bits with probability-based coloring.
 * Shows payload structure: Sync(16) + Timestamp(32) + Auth(64) + CRC(16)
 * Supports real-time highlighting of current bit during playback.
 */

import { useState, useEffect, useRef } from 'react';

// Bit field ranges
const SECTIONS = [
  { name: 'Sync', start: 0, end: 15, color: 'emerald' },
  { name: 'Timestamp', start: 16, end: 47, color: 'blue' },
  { name: 'Auth', start: 48, end: 111, color: 'purple' },
  { name: 'CRC', start: 112, end: 127, color: 'orange' },
];

export function BitMatrixView({
  bitProbs,
  showLabels = true,
  currentBitIndex = -1,
  isProgressive = false,  // Enable playback-synced progressive reveal
  isPlaying = false       // Whether audio is currently playing
}) {
  const [updatedCells, setUpdatedCells] = useState(new Set());
  const prevProbs = useRef(null);

  // Track which cells updated for animation
  useEffect(() => {
    if (!bitProbs || !prevProbs.current) {
      prevProbs.current = bitProbs;
      return;
    }

    const newUpdates = new Set();
    for (let i = 0; i < 128; i++) {
      if (Math.abs(bitProbs[i] - prevProbs.current[i]) > 0.1) {
        newUpdates.add(i);
      }
    }

    setUpdatedCells(newUpdates);
    prevProbs.current = bitProbs ? new Float32Array(bitProbs) : null;

    // Clear animation after delay
    const timer = setTimeout(() => {
      setUpdatedCells(new Set());
    }, 300);

    return () => clearTimeout(timer);
  }, [bitProbs]);

  /**
   * Get section for a bit index
   */
  const getSection = (index) => {
    return SECTIONS.find(s => index >= s.start && index <= s.end);
  };

  /**
   * Get cell color based on probability
   * 0 → Black, 1 → White (Grayscale)
   */
  const getCellColor = (prob) => {
    if (prob === null || prob === undefined) {
      return 'bg-gray-700';
    }

    // Grayscale mapping
    // Prob 0 -> 0% lightness (Black)
    // Prob 1 -> 100% lightness (White)
    // We can use Math.floor(prob * 100)
    const lightness = Math.floor(prob * 100);
    return `hsl(0, 0%, ${lightness}%)`;
  };

  /**
   * Get cell text color based on background lightness
   */
  const getTextColor = (prob) => {
    if (prob === null || prob === undefined) return '#9CA3AF'; // Gray-400
    // If background is dark (prob < 0.5), text should be white
    // If background is light (prob >= 0.5), text should be black
    return prob < 0.5 ? '#FFFFFF' : '#000000';
  };

  /**
   * Get cell value (0 or 1)
   */
  const getCellValue = (prob) => {
    if (prob === null || prob === undefined) return '?';
    return prob > 0.5 ? '1' : '0';
  };

  /**
   * Check if sync pattern is valid (first 16 bits should be 1010101010101010)
   */
  const isSyncValid = () => {
    if (!bitProbs || bitProbs.length < 16) return null;

    const syncPattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let matches = 0;

    for (let i = 0; i < 16; i++) {
      const detected = bitProbs[i] > 0.5 ? 1 : 0;
      if (detected === syncPattern[i]) matches++;
    }

    return matches / 16;
  };

  const syncValidity = isSyncValid();

  // Generate 8x16 grid (128 bits)
  const rows = 8;
  const cols = 16;

  return (
    <div className="glass rounded-xl p-10">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">Bit Matrix</h3>
        <div className="flex items-center gap-2">
          {syncValidity !== null && (
            <span className={`text-xs font-medium px-2 py-0.5 rounded
              ${syncValidity >= 0.9 ? 'bg-green-500/20 text-green-400' :
                syncValidity >= 0.7 ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'}
            `}>
              Sync: {(syncValidity * 100).toFixed(0)}%
            </span>
          )}
          <span className="text-xs text-gray-500">128 bits</span>
        </div>
      </div>


      <div className="flex gap-1">
        {/* Left Sidebar (Vertical Bars) */}
        <div className="w-1 flex flex-col gap-1 py-[1px]">
          {SECTIONS.map((section) => {
            // Calculate relative height based on rows
            // 16 bits per row
            const rowCount = (section.end - section.start + 1) / 16;

            const colorClasses = {
              emerald: 'bg-emerald-500',
              blue: 'bg-blue-500',
              purple: 'bg-purple-500',
              orange: 'bg-orange-500',
            };

            return (
              <div
                key={section.name}
                className={`w-full rounded-full ${colorClasses[section.color]} opacity-90`}
                style={{ flexGrow: rowCount }}
                title={section.name}
              />
            );
          })}
        </div>

        {/* Grid */}
        <div className="flex-grow grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {Array.from({ length: rows * cols }).map((_, index) => {
            const prob = bitProbs ? bitProbs[index] : null;
            const isUpdated = updatedCells.has(index);
            const isCurrent = index === currentBitIndex;

            // Progressive mode: only show bits that have been "scanned"
            const isScanned = !isProgressive || currentBitIndex === -1 || index <= currentBitIndex;
            const showValue = isProgressive ? (isScanned && isPlaying) || currentBitIndex === -1 : true;

            const section = getSection(index);

            // Border color based on section
            const borderColorMap = {
              emerald: 'border-emerald-500/30',
              blue: 'border-blue-500/30',
              purple: 'border-purple-500/30',
              orange: 'border-orange-500/30',
            };
            const borderColor = section ? borderColorMap[section.color] : '';

            // In progressive mode, unscanned bits show gray
            const cellColor = isProgressive && !isScanned && isPlaying
              ? 'bg-gray-700'
              : getCellColor(prob);

            const textColor = isProgressive && !isScanned && isPlaying
              ? '#6b7280'
              : getTextColor(prob);

            return (
              <div
                key={index}
                className={`
                aspect-square rounded-sm flex items-center justify-center
                text-[8px] font-mono font-bold border
                transition-all duration-100
                ${isUpdated ? 'bit-update scale-110' : ''}
                ${prob !== null && isScanned ? 'shadow-sm' : ''}
                ${isCurrent && isPlaying ? 'ring-2 ring-white ring-offset-1 ring-offset-transparent z-10 animate-pulse' : ''}
                ${isProgressive && !isScanned && isPlaying ? 'opacity-40' : ''}
                ${borderColor}
              `}
                style={{
                  backgroundColor: cellColor,
                  color: textColor,
                }}
                title={`[${section?.name}] Bit ${index}: ${prob !== null ? (prob * 100).toFixed(1) + '%' : 'N/A'}`}
              >
                {showLabels ? (isProgressive && !isScanned && isPlaying ? '?' : getCellValue(prob)) : ''}
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm border border-gray-600" style={{ backgroundColor: '#000000' }} />
            <span>0</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#808080' }} />
            <span>?</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm border border-gray-600" style={{ backgroundColor: '#FFFFFF' }} />
            <span>1</span>
          </div>
        </div>

        {/* Section colors legend */}
        <div className="flex items-center gap-2 text-[10px]">
          <span className="text-emerald-400">●Sync</span>
          <span className="text-blue-400">●Time</span>
          <span className="text-purple-400">●Auth</span>
          <span className="text-orange-400">●CRC</span>
        </div>
      </div>
    </div>
  );
}

export default BitMatrixView;
