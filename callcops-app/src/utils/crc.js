/**
 * CRC Utilities for Message Verification
 * 
 * 128-bit payload structure:
 * - 112 bits: data (Sync 16 + Timestamp 32 + Auth 64)
 * - 16 bits: CRC-16 checksum
 * 
 * This allows robust error detection and single-bit correction.
 */

// CRC-16-CCITT polynomial (x^16 + x^12 + x^5 + 1)
const CRC16_POLY = 0x1021;

/**
 * Calculate CRC-16-CCITT checksum (Bit-serial MSB-first)
 * @param {Float32Array|number[]} bits - Array of 0/1 values
 * @returns {number} 16-bit CRC value
 */
export function calculateCRC16(bits) {
  let crc = 0xFFFF; // Initial value

  // Only process the first 112 bits (data portion)
  const len = Math.min(bits.length, 112);

  for (let i = 0; i < len; i++) {
    const bit = bits[i] > 0.5 ? 1 : 0;

    // Process bit MSB-first
    let msb = (crc >> 15) & 1;
    crc = (crc << 1) & 0xFFFF;

    if (msb ^ bit) {
      crc ^= CRC16_POLY;
    }
  }

  return crc;
}

/**
 * Convert bit array to bytes
 * @param {Float32Array|number[]} bits 
 * @returns {Uint8Array}
 */
function bitsToBytes(bits) {
  const numBytes = Math.ceil(bits.length / 8);
  const bytes = new Uint8Array(numBytes);

  for (let i = 0; i < bits.length; i++) {
    const byteIdx = Math.floor(i / 8);
    const bitIdx = 7 - (i % 8);
    if (bits[i] > 0.5) {
      bytes[byteIdx] |= (1 << bitIdx);
    }
  }

  return bytes;
}

/**
 * Convert 16-bit number to bit array
 * @param {number} value 
 * @param {number} length 
 * @returns {number[]}
 */
function numberToBits(value, length) {
  const bits = [];
  for (let i = length - 1; i >= 0; i--) {
    bits.push((value >> i) & 1);
  }
  return bits;
}

/**
 * Extract CRC from 128-bit message
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ data: number[], crc: number }}
 */
export function extractCRC(message) {
  const data = Array.from(message.slice(0, 112)).map(b => b > 0.5 ? 1 : 0);
  const crcBits = Array.from(message.slice(112, 128)).map(b => b > 0.5 ? 1 : 0);

  // Convert 16 bits back to number
  let crc = 0;
  for (let i = 0; i < 16; i++) {
    if (crcBits[i] > 0.5) {
      crc |= (1 << (15 - i));
    }
  }

  return { data, crc };
}

/**
 * Create 128-bit message with CRC
 * @param {Float32Array|number[]} dataBits - 112 bits of data
 * @returns {Float32Array} 128-bit message with CRC appended
 */
export function createMessageWithCRC(dataBits) {
  const message = new Float32Array(128);

  // Copy data bits (Sync + Time + Auth = 112 bits)
  for (let i = 0; i < 112 && i < dataBits.length; i++) {
    message[i] = dataBits[i] > 0.5 ? 1 : 0;
  }

  // Calculate and append 16-bit CRC
  const crc = calculateCRC16(message);
  const crcBits = numberToBits(crc, 16);
  for (let i = 0; i < 16; i++) {
    message[112 + i] = crcBits[i];
  }

  return message;
}

// Expected Sync Pattern: 0xAAAA = 1010101010101010
const SYNC_PATTERN = 0xAAAA;

/**
 * Verify sync pattern (first 16 bits must be 10101010...)
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ isValid: boolean, extractedSync: number }}
 */
export function verifySyncPattern(message) {
  let extractedSync = 0;

  // Extract first 16 bits as a number
  for (let i = 0; i < 16; i++) {
    const bit = message[i] > 0.5 ? 1 : 0;
    extractedSync |= (bit << (15 - i));
  }

  const isValid = extractedSync === SYNC_PATTERN;

  return {
    isValid,
    extractedSync,
    expectedSync: SYNC_PATTERN
  };
}

/**
 * Rotate array to the left by n positions
 * @param {Float32Array|number[]} arr 
 * @param {number} n 
 * @returns {Float32Array}
 */
function rotateLeft(arr, n) {
  const result = new Float32Array(arr.length);
  const len = arr.length;
  for (let i = 0; i < len; i++) {
    result[i] = arr[(i + n) % len];
  }
  return result;
}

/**
 * Attempt to align the message by rotating until Sync Pattern matches
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ alignedMessage: Float32Array, shift: number, found: boolean }}
 */
export function alignBySyncPattern(message) {
  const len = message.length;

  // Try all possible 128 shifts
  for (let shift = 0; shift < len; shift++) {
    const rotated = rotateLeft(message, shift);
    const { isValid } = verifySyncPattern(rotated);

    if (isValid) {
      return { alignedMessage: rotated, shift, found: true };
    }
  }

  return { alignedMessage: message, shift: 0, found: false };
}

/**
 * Verify message integrity using CRC AND Sync Pattern
 * Automatically attempts to align the message if Sync Pattern is displaced.
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ isValid: boolean, expectedCRC: number, actualCRC: number, confidence: number, syncValid: boolean, shift?: number }}
 */
export function verifyCRC(message) {
  // 1. First, try to align based on Sync Pattern
  const { alignedMessage, shift, found } = alignBySyncPattern(message);

  // 2. Use the aligned message for CRC check
  const { crc: actualCRC } = extractCRC(alignedMessage);
  // Re-calculate expected CRC from the data portion
  const expectedCRC = calculateCRC16(alignedMessage);

  const crcValid = expectedCRC === actualCRC;

  // If we found a sync pattern AND the CRC matches, it's a valid message!
  const isValid = found && crcValid;

  // Calculate average bit confidence
  const confidences = Array.from(alignedMessage).map(b => Math.abs(b - 0.5) * 2);
  const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;

  return {
    isValid,
    expectedCRC,
    actualCRC,
    confidence: avgConfidence,
    syncValid: found,
    extractedSync: found ? SYNC_PATTERN : 0, // Simplified for success case
    shift // Return shift amount for debugging
  };
}

/**
 * Find suspicious bits based on low confidence
 * @param {Float32Array|number[]} probs - Probability values (0-1)
 * @param {number} threshold - Confidence threshold (default 0.7)
 * @returns {number[]} Indices of suspicious bits
 */
export function findSuspiciousBits(probs, threshold = 0.7) {
  const suspicious = [];

  for (let i = 0; i < probs.length; i++) {
    const confidence = Math.abs(probs[i] - 0.5) * 2;
    if (confidence < threshold) {
      suspicious.push({
        index: i,
        prob: probs[i],
        confidence
      });
    }
  }

  return suspicious.sort((a, b) => a.confidence - b.confidence);
}

/**
 * Attempt single-bit error correction
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ corrected: Float32Array, correctedBit: number|null, success: boolean }}
 */
export function attemptCorrection(message) {
  const { isValid } = verifyCRC(message);

  if (isValid) {
    return {
      corrected: new Float32Array(message),
      correctedBit: null,
      success: true
    };
  }

  // Find low-confidence bits and try flipping them
  const suspicious = findSuspiciousBits(message, 0.8);

  for (const { index } of suspicious) {
    const trial = new Float32Array(message);
    trial[index] = trial[index] > 0.5 ? 0 : 1;

    const { isValid: nowValid } = verifyCRC(trial);
    if (nowValid) {
      return {
        corrected: trial,
        correctedBit: index,
        success: true
      };
    }
  }

  return {
    corrected: new Float32Array(message),
    correctedBit: null,
    success: false
  };
}

/**
 * Format message as hex string
 * @param {Float32Array|number[]} message 
 * @returns {string}
 */
export function messageToHex(message) {
  const bytes = bitsToBytes(message);
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Calculate similarity between two messages
 * @param {Float32Array|number[]} msg1 
 * @param {Float32Array|number[]} msg2 
 * @returns {{ matchingBits: number, totalBits: number, accuracy: number }}
 */
export function compareMessages(msg1, msg2) {
  const len = Math.min(msg1.length, msg2.length);
  let matching = 0;

  for (let i = 0; i < len; i++) {
    const b1 = msg1[i] > 0.5 ? 1 : 0;
    const b2 = msg2[i] > 0.5 ? 1 : 0;
    if (b1 === b2) matching++;
  }

  return {
    matchingBits: matching,
    totalBits: len,
    accuracy: matching / len
  };
}

export default {
  calculateCRC16,
  createMessageWithCRC,
  verifyCRC,
  findSuspiciousBits,
  attemptCorrection,
  messageToHex,
  compareMessages
};
