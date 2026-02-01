/**
 * CRC Utilities for Message Verification
 * 
 * 128-bit payload structure:
 * - 120 bits: data
 * - 8 bits: CRC-8 checksum
 * 
 * This allows error detection and potential correction.
 */

// CRC-8 polynomial (x^8 + x^2 + x + 1)
const CRC8_POLY = 0x07;

/**
 * Calculate CRC-8 checksum
 * @param {Float32Array|number[]} bits - Array of 0/1 values (120 bits for data)
 * @returns {number} 8-bit CRC value
 */
export function calculateCRC8(bits) {
  const data = bitsToBytes(bits.slice(0, 120));
  let crc = 0x00;
  
  for (const byte of data) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) {
      if (crc & 0x80) {
        crc = ((crc << 1) ^ CRC8_POLY) & 0xFF;
      } else {
        crc = (crc << 1) & 0xFF;
      }
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
 * Convert byte to bit array
 * @param {number} byte 
 * @returns {number[]}
 */
function byteToBits(byte) {
  const bits = [];
  for (let i = 7; i >= 0; i--) {
    bits.push((byte >> i) & 1);
  }
  return bits;
}

/**
 * Extract CRC from 128-bit message
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ data: number[], crc: number }}
 */
export function extractCRC(message) {
  const data = Array.from(message.slice(0, 120)).map(b => b > 0.5 ? 1 : 0);
  const crcBits = Array.from(message.slice(120, 128)).map(b => b > 0.5 ? 1 : 0);
  const crc = crcBits.reduce((acc, bit, i) => acc | (bit << (7 - i)), 0);
  
  return { data, crc };
}

/**
 * Create 128-bit message with CRC
 * @param {Float32Array|number[]} dataBits - 120 bits of data
 * @returns {Float32Array} 128-bit message with CRC appended
 */
export function createMessageWithCRC(dataBits) {
  const message = new Float32Array(128);
  
  // Copy data bits
  for (let i = 0; i < 120 && i < dataBits.length; i++) {
    message[i] = dataBits[i] > 0.5 ? 1 : 0;
  }
  
  // Calculate and append CRC
  const crc = calculateCRC8(message);
  const crcBits = byteToBits(crc);
  for (let i = 0; i < 8; i++) {
    message[120 + i] = crcBits[i];
  }
  
  return message;
}

/**
 * Verify message integrity using CRC
 * @param {Float32Array|number[]} message - 128-bit message
 * @returns {{ isValid: boolean, expectedCRC: number, actualCRC: number, confidence: number }}
 */
export function verifyCRC(message) {
  const { data, crc: actualCRC } = extractCRC(message);
  const expectedCRC = calculateCRC8(message);
  
  const isValid = expectedCRC === actualCRC;
  
  // Calculate bit confidence (how far from 0.5)
  const confidences = Array.from(message).map(b => Math.abs(b - 0.5) * 2);
  const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
  
  return {
    isValid,
    expectedCRC,
    actualCRC,
    confidence: avgConfidence
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
  calculateCRC8,
  createMessageWithCRC,
  verifyCRC,
  findSuspiciousBits,
  attemptCorrection,
  messageToHex,
  compareMessages
};
