import React from 'react';

/**
 * CallResultScreen - Displays verification result after analyzing audio
 * Shows caller info if valid, or scam warning if invalid
 */
export default function CallResultScreen({ isValid, crcValid, payload, onEndCall }) {
    // Parse payload bits into meaningful sections (if available)
    const parsePayload = () => {
        if (!payload || payload.length < 128) {
            return { sync: '—', timestamp: '—', authCode: '—', rs: '—' };
        }

        // Extract bit sections (RS structure: 96 data + 32 parity)
        const syncBits = payload.slice(0, 16);
        const timestampBits = payload.slice(16, 48);
        const authBits = payload.slice(48, 96);  // 48 bits (was 64)
        const rsBits = payload.slice(96, 128);   // 32 bits RS parity (was 16 CRC)

        // Convert to values
        const syncValue = bitsToHex(syncBits);
        const timestampValue = bitsToInt(timestampBits);
        const authValue = bitsToHex(authBits.slice(0, 24)) + '...'; // Partial display
        const rsValue = bitsToHex(rsBits);

        // Format timestamp (assuming Unix timestamp in seconds)
        const date = new Date(timestampValue * 1000);
        const formattedTime = isNaN(date.getTime()) ? '—' : date.toLocaleString('ko-KR');

        return {
            sync: syncValue,
            timestamp: formattedTime,
            authCode: authValue,
            rs: rsValue
        };
    };

    const bitsToInt = (bits) => {
        return bits.reduce((acc, bit, i) => acc + (bit ? Math.pow(2, bits.length - 1 - i) : 0), 0);
    };

    const bitsToHex = (bits) => {
        const num = bitsToInt(bits);
        return '0x' + num.toString(16).toUpperCase().padStart(Math.ceil(bits.length / 4), '0');
    };

    const info = parsePayload();

    return (
        <div className={`min-h-screen flex flex-col items-center justify-between py-12 px-6 ${isValid
                ? 'bg-gradient-to-b from-green-900/50 via-gray-900 to-gray-900'
                : 'bg-gradient-to-b from-red-900/50 via-gray-900 to-gray-900'
            }`}>
            {/* Status Icon */}
            <div className="flex-1 flex flex-col items-center justify-center">
                <div className={`w-28 h-28 rounded-full flex items-center justify-center mb-6 ${isValid
                        ? 'bg-green-500/20 border-4 border-green-500'
                        : 'bg-red-500/20 border-4 border-red-500'
                    }`}>
                    {isValid ? (
                        <svg className="w-14 h-14 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                    ) : (
                        <svg className="w-14 h-14 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                    )}
                </div>

                {/* Status Title */}
                <h1 className={`text-3xl font-bold mb-2 ${isValid ? 'text-green-400' : 'text-red-400'}`}>
                    {isValid ? 'Verified Caller' : 'Spoofed Call'}
                </h1>
                <p className={`text-lg mb-8 ${isValid ? 'text-green-300/70' : 'text-red-300/70'}`}>
                    {isValid ? 'This call is authenticated' : 'Warning: Potential Voice Phishing'}
                </p>

                {/* Caller Info Card (shown only for valid calls) */}
                {isValid && (
                    <div className="w-full max-w-sm bg-gray-800/50 rounded-2xl p-5 border border-gray-700/50 space-y-4">
                        <h2 className="text-gray-400 text-sm font-medium uppercase tracking-wider mb-3">Caller Information</h2>

                        <div className="flex justify-between items-center">
                            <span className="text-gray-500">Sync Pattern</span>
                            <span className="text-white font-mono text-sm">{info.sync}</span>
                        </div>

                        <div className="flex justify-between items-center">
                            <span className="text-gray-500">Timestamp</span>
                            <span className="text-white font-mono text-sm">{info.timestamp}</span>
                        </div>

                        <div className="flex justify-between items-center">
                            <span className="text-gray-500">Auth Code</span>
                            <span className="text-white font-mono text-sm">{info.authCode}</span>
                        </div>

                        <div className="flex justify-between items-center">
                            <span className="text-gray-500">RS Check</span>
                            <span className={`font-mono text-sm ${crcValid ? 'text-green-400' : 'text-yellow-400'}`}>
                                {crcValid ? '✓ Valid' : '⚠ Error Detected'}
                            </span>
                        </div>
                    </div>
                )}

                {/* Warning Card (shown only for invalid calls) */}
                {!isValid && (
                    <div className="w-full max-w-sm bg-red-900/20 rounded-2xl p-5 border border-red-700/30 space-y-3">
                        <div className="flex items-center gap-2 text-red-400 font-semibold">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01" />
                            </svg>
                            Security Warning
                        </div>
                        <p className="text-gray-300 text-sm leading-relaxed">
                            This audio does not contain a valid authentication watermark.
                            The caller may be attempting to impersonate a trusted source.
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1 mt-3">
                            <li>• Do not share personal information</li>
                            <li>• Do not transfer money</li>
                            <li>• Report suspicious calls to 112</li>
                        </ul>
                    </div>
                )}
            </div>

            {/* End Call Button */}
            <button
                onClick={onEndCall}
                className="w-20 h-20 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-lg shadow-red-500/30 hover:scale-110 transition-transform active:scale-95 mt-8"
            >
                <svg className="w-10 h-10 text-white rotate-[135deg]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
            </button>
            <span className="text-red-400 text-sm font-medium mt-3">End Call</span>
        </div>
    );
}
