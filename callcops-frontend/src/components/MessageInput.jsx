/**
 * MessageInput Component
 * 
 * Input for 128-bit watermark message matching CallCops payload structure with Reed-Solomon:
 * - [0-15]   Sync Pattern (16 bits) - Fixed: 1010101010101010
 * - [16-47]  Timestamp (32 bits) - Random/User input
 * - [48-95]  Auth Data (48 bits) - User input
 * - [96-127] RS Parity (32 bits) - Auto-calculated (can correct up to 2 byte errors)
 */

import { useState, useCallback, useEffect } from 'react';
import { createMessageWithRS } from '../utils/reedSolomon';

// Fixed sync pattern: 1010101010101010
const SYNC_PATTERN = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

// Bit field definitions (RS structure)
const SYNC_BITS = 16;
const TIMESTAMP_BITS = 32;
const AUTH_BITS = 48;  // Reduced from 64 to make room for RS parity
const RS_PARITY_BITS = 32;  // RS(16,12) parity for 2-byte error correction

export function MessageInput({ message, onChange, disabled = false }) {
  // Store raw hex strings for display (allowing partial input)
  const [timestampHex, setTimestampHex] = useState('');
  const [authHex, setAuthHex] = useState('');

  // Create full message with RS parity (using high-level API)
  const createFullMessage = useCallback((dataBits) => {
    // dataBits should be 96 bits: Sync(16) + Time(32) + Auth(48)
    return createMessageWithRS(dataBits);
  }, []);

  // Convert hex string to bits (pad with zeros on right if needed)
  const hexToBits = useCallback((hex, targetLength) => {
    const bits = [];
    const cleanHex = hex.replace(/[^0-9A-Fa-f]/g, '');
    for (let i = 0; i < cleanHex.length; i++) {
      const nibble = parseInt(cleanHex[i], 16);
      bits.push((nibble >> 3) & 1);
      bits.push((nibble >> 2) & 1);
      bits.push((nibble >> 1) & 1);
      bits.push(nibble & 1);
    }
    // Pad with zeros if needed
    while (bits.length < targetLength) {
      bits.push(0);
    }
    return bits.slice(0, targetLength);
  }, []);

  // Convert bits to hex string
  const bitsToHex = useCallback((bits) => {
    if (!bits || bits.length === 0) return '';
    let hex = '';
    for (let i = 0; i < bits.length; i += 4) {
      const nibble = (bits[i] || 0) * 8 + (bits[i + 1] || 0) * 4 + (bits[i + 2] || 0) * 2 + (bits[i + 3] || 0);
      hex += nibble.toString(16).toUpperCase();
    }
    return hex;
  }, []);

  // Build full 128-bit message from hex strings with RS parity
  const buildMessage = useCallback(() => {
    // Both fields must have at least some input
    if (!timestampHex && !authHex) return null;

    const sync = [...SYNC_PATTERN];
    const timestamp = hexToBits(timestampHex, TIMESTAMP_BITS);
    const auth = hexToBits(authHex, AUTH_BITS);
    const dataBits = [...sync, ...timestamp, ...auth]; // 96 bits
    
    // Use RS to create full 128-bit message with parity
    return createFullMessage(dataBits);
  }, [timestampHex, authHex, hexToBits, createFullMessage]);

  // Update parent when hex strings change
  useEffect(() => {
    const msg = buildMessage();
    if (msg) {
      onChange(msg);
    }
  }, [timestampHex, authHex, buildMessage, onChange]);

  // Handle timestamp input change
  const handleTimestampChange = useCallback((e) => {
    const value = e.target.value.replace(/[^0-9A-Fa-f]/g, '').toUpperCase();
    setTimestampHex(value.slice(0, 8)); // Max 8 hex chars = 32 bits
  }, []);

  // Handle auth data input change
  const handleAuthChange = useCallback((e) => {
    const value = e.target.value.replace(/[^0-9A-Fa-f]/g, '').toUpperCase();
    setAuthHex(value.slice(0, 12)); // Max 12 hex chars = 48 bits
  }, []);

  // Generate random timestamp
  const generateRandomTimestamp = useCallback(() => {
    const bits = Array.from({ length: TIMESTAMP_BITS }, () => Math.random() > 0.5 ? 1 : 0);
    setTimestampHex(bitsToHex(bits));
  }, [bitsToHex]);

  // Generate random auth data
  const generateRandomAuth = useCallback(() => {
    const bits = Array.from({ length: AUTH_BITS }, () => Math.random() > 0.5 ? 1 : 0);
    setAuthHex(bitsToHex(bits));
  }, [bitsToHex]);

  // Generate all random
  const generateRandomAll = useCallback(() => {
    generateRandomTimestamp();
    generateRandomAuth();
  }, [generateRandomTimestamp, generateRandomAuth]);

  // Calculate RS parity for display (based on current input)
  const displayRS = (timestampHex || authHex)
    ? (() => {
        const sync = [...SYNC_PATTERN];
        const timestamp = hexToBits(timestampHex, TIMESTAMP_BITS);
        const auth = hexToBits(authHex, AUTH_BITS);
        const dataBits = [...sync, ...timestamp, ...auth];
        const fullMsg = createFullMessage(dataBits);
        // RS parity is the last 32 bits (4 bytes)
        const parityBits = Array.from(fullMsg.slice(96, 128));
        // Convert to hex
        let hex = '';
        for (let i = 0; i < 32; i += 4) {
          const nibble = parityBits[i] * 8 + parityBits[i+1] * 4 + parityBits[i+2] * 2 + parityBits[i+3];
          hex += nibble.toString(16).toUpperCase();
        }
        return hex;
      })()
    : '--------';

  // Display filled timestamp/auth for visual feedback
  const displayTimestamp = timestampHex.padEnd(8, '·');
  const displayAuth = authHex.padEnd(12, '·');  // 12 hex chars for 48 bits

  return (
    <div className="space-y-4">
      {/* Sync Pattern (Fixed) */}
      <div className="bg-surface/50 rounded-lg p-3 border border-gray-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">Sync Pattern (16 bits)</span>
          <span className="text-xs text-green-400">Fixed</span>
        </div>
        <div className="font-mono text-sm text-green-400 tracking-wider">
          1010 1010 1010 1010
        </div>
      </div>

      {/* Timestamp (32 bits) */}
      <div className="bg-surface/50 rounded-lg p-3 border border-gray-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">Timestamp (32 bits)</span>
          <button
            onClick={generateRandomTimestamp}
            disabled={disabled}
            className="text-xs text-purple-400 hover:text-purple-300 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Random
          </button>
        </div>
        <div className="relative">
          <input
            type="text"
            value={timestampHex}
            onChange={handleTimestampChange}
            disabled={disabled}
            placeholder="Enter 8 hex chars..."
            maxLength={8}
            className="w-full bg-transparent font-mono text-sm text-gray-200 tracking-wider
                     placeholder:text-gray-600 outline-none
                     disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {/* Character count indicator */}
          <span className="absolute right-0 top-0 text-xs text-gray-500">
            {timestampHex.length}/8
          </span>
        </div>
      </div>

      {/* Auth Data (48 bits) */}
      <div className="bg-surface/50 rounded-lg p-3 border border-gray-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">Auth Data (48 bits)</span>
          <button
            onClick={generateRandomAuth}
            disabled={disabled}
            className="text-xs text-purple-400 hover:text-purple-300 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Random
          </button>
        </div>
        <div className="relative">
          <input
            type="text"
            value={authHex}
            onChange={handleAuthChange}
            disabled={disabled}
            placeholder="Enter 12 hex chars..."
            maxLength={12}
            className="w-full bg-transparent font-mono text-sm text-gray-200 tracking-wider
                     placeholder:text-gray-600 outline-none
                     disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {/* Character count indicator */}
          <span className="absolute right-0 top-0 text-xs text-gray-500">
            {authHex.length}/12
          </span>
        </div>
      </div>

      {/* RS Parity (Auto-calculated) */}
      <div className="bg-surface/50 rounded-lg p-3 border border-gray-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">RS Parity (32 bits)</span>
          <span className="text-xs text-blue-400">Auto · 2-byte ECC</span>
        </div>
        <div className="font-mono text-sm text-blue-400 tracking-wider">
          {displayRS}
        </div>
      </div>

      {/* Generate All Button */}
      <button
        onClick={generateRandomAll}
        disabled={disabled}
        className="w-full py-2.5 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 
                 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2
                 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        Generate Random Payload
      </button>

      {/* Payload Structure Info */}
      <div className="text-xs text-gray-500 text-center">
        Total: 128 bits = Sync(16) + Timestamp(32) + Auth(48) + RS(32)
      </div>
    </div>
  );
}

export default MessageInput;
