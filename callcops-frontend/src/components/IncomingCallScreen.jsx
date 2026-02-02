import React from 'react';

/**
 * IncomingCallScreen - Displays an incoming call UI with ringing animation
 * Shows "Unknown Caller" and provides Answer/Decline buttons
 */
export default function IncomingCallScreen({ onAnswer, onDecline }) {
    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 flex flex-col items-center justify-between py-16 px-6">
            {/* Caller Info Section */}
            <div className="flex-1 flex flex-col items-center justify-center">
                {/* Pulsing Avatar */}
                <div className="relative mb-8">
                    <div className="absolute inset-0 rounded-full bg-green-500/20 animate-ping" style={{ animationDuration: '2s' }} />
                    <div className="absolute inset-0 rounded-full bg-green-500/10 animate-ping" style={{ animationDuration: '2s', animationDelay: '0.5s' }} />
                    <div className="relative w-32 h-32 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center border-4 border-gray-600 shadow-2xl">
                        <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                    </div>
                </div>

                {/* Caller Name */}
                <h1 className="text-3xl font-bold text-white mb-2">Unknown Caller</h1>
                <p className="text-gray-400 text-lg mb-4">Incoming Call</p>

                {/* Ringing Indicator */}
                <div className="flex items-center gap-2 text-green-400">
                    <span className="inline-block w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <span className="text-sm font-medium">Ringing...</span>
                </div>

                {/* Security Notice */}
                <div className="mt-8 px-4 py-3 bg-yellow-500/10 border border-yellow-500/30 rounded-xl max-w-xs">
                    <p className="text-yellow-400 text-center text-sm">
                        📁 Tap "Answer" to upload audio file for verification
                    </p>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-center gap-16">
                {/* Decline Button */}
                <button
                    onClick={onDecline}
                    className="w-20 h-20 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-lg shadow-red-500/30 hover:scale-110 transition-transform active:scale-95"
                >
                    <svg className="w-10 h-10 text-white rotate-[135deg]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                    </svg>
                </button>

                {/* Answer Button */}
                <button
                    onClick={onAnswer}
                    className="w-20 h-20 rounded-full bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center shadow-lg shadow-green-500/30 hover:scale-110 transition-transform active:scale-95 animate-pulse"
                    style={{ animationDuration: '1.5s' }}
                >
                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                    </svg>
                </button>
            </div>

            {/* Labels */}
            <div className="flex items-center justify-center gap-16 mt-4">
                <span className="text-red-400 text-sm font-medium w-20 text-center">Decline</span>
                <span className="text-green-400 text-sm font-medium w-20 text-center">Answer</span>
            </div>
        </div>
    );
}
