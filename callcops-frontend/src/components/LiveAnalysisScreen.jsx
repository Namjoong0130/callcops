import React from 'react';

/**
 * LiveAnalysisScreen - Shows real-time detection progress
 * Displays bit probabilities, RS status, and validation state as they update
 */
export default function LiveAnalysisScreen({
    progress,
    bitProbs,
    currentFrame,
    totalFrames,
    crcValid,
    isValid,
    confidence,
    fileName
}) {
    // Convert bit probs to visual representation
    const renderBitMatrix = () => {
        if (!bitProbs || bitProbs.length === 0) {
            return (
                <div className="grid grid-cols-16 gap-0.5">
                    {Array(128).fill(0).map((_, i) => (
                        <div key={i} className="w-4 h-4 bg-gray-700 rounded-sm" />
                    ))}
                </div>
            );
        }

        return (
            <div className="grid grid-cols-16 gap-0.5">
                {bitProbs.map((prob, i) => {
                    const isOne = prob > 0.5;
                    const strength = Math.abs(prob - 0.5) * 2; // 0-1 range
                    const opacity = 0.3 + strength * 0.7;

                    return (
                        <div
                            key={i}
                            className={`w-4 h-4 rounded-sm transition-all duration-150 ${isOne ? 'bg-green-500' : 'bg-red-500'
                                }`}
                            style={{ opacity }}
                            title={`Bit ${i}: ${(prob * 100).toFixed(1)}%`}
                        />
                    );
                })}
            </div>
        );
    };

    // Status indicator component
    const StatusBadge = ({ label, value, isGood }) => (
        <div className="flex items-center justify-between px-3 py-2 bg-gray-800/50 rounded-lg">
            <span className="text-gray-400 text-sm">{label}</span>
            <span className={`font-mono text-sm font-bold ${value === null ? 'text-gray-500' :
                    isGood ? 'text-green-400' : 'text-red-400'
                }`}>
                {value === null ? '...' : value}
            </span>
        </div>
    );

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 flex flex-col items-center py-8 px-4">
            {/* Header */}
            <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-white mb-1">Analyzing Call</h2>
                <p className="text-gray-400 text-sm">{fileName || 'Audio file'}</p>
            </div>

            {/* Waveform-like progress indicator */}
            <div className="w-full max-w-sm mb-6">
                <div className="flex items-center gap-1 h-12 justify-center">
                    {Array(20).fill(0).map((_, i) => {
                        const isActive = (i / 20) <= (progress / 100);
                        const height = 8 + Math.sin(i * 0.5 + Date.now() / 200) * 8;
                        return (
                            <div
                                key={i}
                                className={`w-2 rounded-full transition-all duration-300 ${isActive ? 'bg-blue-500' : 'bg-gray-700'
                                    }`}
                                style={{ height: `${height}px` }}
                            />
                        );
                    })}
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>Frame {currentFrame || 0}</span>
                    <span>{progress}%</span>
                    <span>{totalFrames || '?'} frames</span>
                </div>
            </div>

            {/* Live Bit Matrix */}
            <div className="w-full max-w-sm mb-6">
                <h3 className="text-gray-400 text-sm font-medium mb-2 text-center">128-Bit Payload Detection</h3>
                <div className="bg-gray-800/30 rounded-xl p-3 border border-gray-700/50">
                    {renderBitMatrix()}
                </div>
                <div className="flex justify-center gap-4 mt-2 text-xs text-gray-500">
                    <span className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-green-500 rounded-full" /> 1
                    </span>
                    <span className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-red-500 rounded-full" /> 0
                    </span>
                    <span className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-gray-600 rounded-full" /> Unknown
                    </span>
                </div>
            </div>

            {/* Live Status Indicators */}
            <div className="w-full max-w-sm space-y-2 mb-6">
                <StatusBadge
                    label="Confidence"
                    value={confidence !== null ? `${(confidence * 100).toFixed(1)}%` : null}
                    isGood={confidence > 0.3}
                />
                <StatusBadge
                    label="RS(16,12) Check"
                    value={crcValid === null ? null : (crcValid ? '✓ Valid' : '✗ Error Detected')}
                    isGood={crcValid}
                />
                <StatusBadge
                    label="Authentication"
                    value={isValid === null ? null : (isValid ? '✓ Verified' : '✗ Spoofed')}
                    isGood={isValid}
                />
            </div>

            {/* Pulsing indicator */}
            <div className="flex items-center gap-2 text-blue-400">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span className="text-sm">Processing...</span>
            </div>
        </div>
    );
}
