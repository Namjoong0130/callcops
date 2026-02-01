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
   * 0 → Red, 0.5 → Yellow, 1 → Green
   */
  const getCellColor = (prob) => {
    if (prob === null || prob === undefined) {
      return 'bg-gray-700';
    }
    
    const hue = prob * 120; // 0=red(0°), 0.5=yellow(60°), 1=green(120°)
    const saturation = 70 + Math.abs(prob - 0.5) * 60; // More saturated at extremes
    const lightness = 40 + Math.abs(prob - 0.5) * 20;
    
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
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
    <div className="glass rounded-xl p-4">
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
      
      {/* Section Labels */}
      <div className="flex mb-2 text-[10px] font-medium">
        {SECTIONS.map((section) => {
          const width = ((section.end - section.start + 1) / 128) * 100;
          const colorClasses = {
            emerald: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
            blue: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
            purple: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
            orange: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
          };
          return (
            <div
              key={section.name}
              className={`px-1 py-0.5 text-center border rounded-sm ${colorClasses[section.color]}`}
              style={{ width: `${width}%` }}
            >
              {section.name}
            </div>
          );
        })}
      </div>
      
      {/* Grid */}
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
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
            : (prob !== null ? (prob > 0.5 ? '#064e3b' : '#7f1d1d') : '#6b7280');
          
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
      
      {/* Legend */}
      <div className="mt-3 flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(0, 100%, 50%)' }} />
            <span>0</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(60, 70%, 50%)' }} />
            <span>?</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(120, 100%, 50%)' }} />
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
