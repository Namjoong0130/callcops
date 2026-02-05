/**
 * Reed-Solomon Error Correction for CallCops
 * 
 * Uses `reedsolomon` npm package (ported from ZXing)
 * 
 * RS(16,12) over GF(2^8) with BIT INTERLEAVING:
 * - 16 bytes codeword = 128 bits
 * - 12 bytes data = 96 bits  
 * - 4 bytes parity = 32 bits
 * - Can correct up to 2 byte errors (16 bits) ANYWHERE in the codeword
 * 
 * INTERLEAVING STRATEGY:
 * Instead of sequential bit-to-byte mapping (bits 0-7 → byte 0),
 * we use strided mapping so that consecutive bit errors spread across bytes:
 *   Byte 0: bits 0, 16, 32, 48, 64, 80, 96, 112
 *   Byte 1: bits 1, 17, 33, 49, 65, 81, 97, 113
 *   ...
 * This way, 16 consecutive bit errors affect at most 2 bytes → correctable!
 * 
 * Message structure:
 * ┌─────────┬───────────┬──────────┬─────────┐
 * │ Sync 16 │ Time 32   │ Auth 48  │ RS 32   │
 * └─────────┴───────────┴──────────┴─────────┘
 *      96 bits data          32 bits parity
 */

// CommonJS module - use require
import * as rsModule from 'reedsolomon';

// Handle both ESM and CommonJS imports
const rs = rsModule.default || rsModule;
const GenericGF = rs.GenericGF || rsModule.GenericGF;
const ReedSolomonEncoder = rs.ReedSolomonEncoder || rsModule.ReedSolomonEncoder;
const ReedSolomonDecoder = rs.ReedSolomonDecoder || rsModule.ReedSolomonDecoder;

// =============================================================================
// RS Parameters
// =============================================================================

const NSYM = 4;       // Number of parity bytes (32 bits) - corrects up to 2 byte errors
const NDATA = 12;     // Number of data bytes (96 bits)
const NCODEWORD = 16; // Total codeword length (128 bits)

// Use GF(2^8) suitable for byte-level data
const gf = GenericGF.QR_CODE_FIELD_256();
const encoder = new ReedSolomonEncoder(gf);
const decoder = new ReedSolomonDecoder(gf);

// =============================================================================
// Core RS Functions
// =============================================================================

/**
 * Encode data bytes with Reed-Solomon parity
 * @param {Uint8Array} data - 12 bytes of data
 * @returns {Uint8Array} - 16 bytes codeword (12 data + 4 parity)
 */
export function rsEncode(data) {
  if (data.length !== NDATA) {
    throw new Error(`Expected ${NDATA} bytes, got ${data.length}`);
  }

  // Create message array with space for parity
  const message = new Int32Array(NCODEWORD);
  for (let i = 0; i < NDATA; i++) {
    message[i] = data[i];
  }

  // Encode in-place
  encoder.encode(message, NSYM);

  // Convert back to Uint8Array
  const codeword = new Uint8Array(NCODEWORD);
  for (let i = 0; i < NCODEWORD; i++) {
    codeword[i] = message[i] & 0xFF;
  }

  return codeword;
}

/**
 * Decode and correct Reed-Solomon codeword
 * @param {Uint8Array} received - 16 bytes received codeword
 * @returns {{ data: Uint8Array, corrected: number, success: boolean }}
 */
export function rsDecode(received) {
  if (received.length !== NCODEWORD) {
    throw new Error(`Expected ${NCODEWORD} bytes, got ${received.length}`);
  }

  // Convert to Int32Array for the decoder
  const message = new Int32Array(NCODEWORD);
  for (let i = 0; i < NCODEWORD; i++) {
    message[i] = received[i];
  }

  try {
    // Decode in-place (throws on uncorrectable error)
    decoder.decode(message, NSYM);

    // Extract corrected data
    const data = new Uint8Array(NDATA);
    for (let i = 0; i < NDATA; i++) {
      data[i] = message[i] & 0xFF;
    }

    // Count corrected bytes by comparing with original
    let corrected = 0;
    for (let i = 0; i < NCODEWORD; i++) {
      if ((message[i] & 0xFF) !== received[i]) {
        corrected++;
      }
    }

    return {
      data,
      corrected,
      success: true
    };
  } catch (e) {
    // Uncorrectable error
    return {
      data: received.slice(0, NDATA),
      corrected: 0,
      success: false
    };
  }
}

// =============================================================================
// Bit Interleaving (spreads consecutive bit errors across different bytes)
// =============================================================================

/**
 * Interleave bits: consecutive bits → spread across bytes
 * bit[i] → byte[i % 16], bit position [floor(i / 16)]
 * 
 * Example for 128 bits → 16 bytes:
 *   Byte 0 gets bits: 0, 16, 32, 48, 64, 80, 96, 112
 *   Byte 1 gets bits: 1, 17, 33, 49, 65, 81, 97, 113
 *   ...
 * 
 * This ensures consecutive bit errors (e.g., bits 0-15) affect different bytes,
 * maximizing RS correction capability.
 */
function interleaveBits(bits) {
  const interleaved = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    // Original bit i goes to interleaved position
    const byteIdx = i % 16;           // Which byte (0-15)
    const bitInByte = Math.floor(i / 16); // Which bit in that byte (0-7)
    const newIdx = byteIdx * 8 + bitInByte;
    interleaved[newIdx] = bits[i];
  }
  return interleaved;
}

/**
 * Deinterleave bits: reverse the interleaving
 */
function deinterleaveBits(interleaved) {
  const original = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    const byteIdx = i % 16;
    const bitInByte = Math.floor(i / 16);
    const newIdx = byteIdx * 8 + bitInByte;
    original[i] = interleaved[newIdx];
  }
  return original;
}

// =============================================================================
// Bit-level Helpers (128 bits ↔ 16 bytes) - WITH INTERLEAVING
// =============================================================================

/**
 * Convert 128-bit array to 16 bytes (NO interleaving for now - model not trained with it)
 */
export function bitsToBytes(bits) {
  // NOTE: Interleaving disabled because the model was not trained with interleaved data
  // If you want interleaving, uncomment the line below:
  // const interleaved = interleaveBits(bits);
  const interleaved = bits; // No interleaving
  
  const bytes = new Uint8Array(16);
  for (let i = 0; i < 128; i++) {
    const byteIdx = Math.floor(i / 8);
    const bitIdx = 7 - (i % 8);
    if (interleaved[i] > 0.5) {
      bytes[byteIdx] |= (1 << bitIdx);
    }
  }
  return bytes;
}

/**
 * Convert 16 bytes to 128-bit Float32Array (NO deinterleaving for now)
 */
export function bytesToBits(bytes) {
  const interleaved = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    const byteIdx = Math.floor(i / 8);
    const bitIdx = 7 - (i % 8);
    interleaved[i] = (bytes[byteIdx] >> bitIdx) & 1;
  }
  // NOTE: Deinterleaving disabled because the model was not trained with interleaved data
  // If you want deinterleaving, uncomment the line below:
  // return deinterleaveBits(interleaved);
  return interleaved; // No deinterleaving
}

/**
 * Convert 96 data bits to 12 bytes (NO interleaving for now)
 */
export function dataBitsToBytes(bits) {
  // NOTE: Interleaving disabled - model not trained with it
  // Original interleaving code commented out below
  /*
  const interleaved = new Float32Array(96);
  for (let i = 0; i < 96; i++) {
    const byteIdx = i % 12;
    const bitInByte = Math.floor(i / 12);
    const newIdx = byteIdx * 8 + bitInByte;
    interleaved[newIdx] = bits[i];
  }
  */
  const interleaved = bits; // No interleaving
  
  const bytes = new Uint8Array(12);
  for (let i = 0; i < 96; i++) {
    const byteIdx = Math.floor(i / 8);
    const bitIdx = 7 - (i % 8);
    if (interleaved[i] > 0.5) {
      bytes[byteIdx] |= (1 << bitIdx);
    }
  }
  return bytes;
}

/**
 * Convert 12 bytes to 96 data bits (NO deinterleaving for now)
 */
export function dataBytesToBits(bytes) {
  // First extract bits from bytes
  const interleaved = new Float32Array(96);
  for (let i = 0; i < 96; i++) {
    const byteIdx = Math.floor(i / 8);
    const bitIdx = 7 - (i % 8);
    interleaved[i] = (bytes[byteIdx] >> bitIdx) & 1;
  }
  
  // NOTE: Deinterleaving disabled - model not trained with it
  // Original deinterleaving code commented out below
  /*
  const bits = new Float32Array(96);
  for (let i = 0; i < 96; i++) {
    const byteIdx = i % 12;
    const bitInByte = Math.floor(i / 12);
    const newIdx = byteIdx * 8 + bitInByte;
    bits[i] = interleaved[newIdx];
  }
  return bits;
  */
  return interleaved; // No deinterleaving
}

// =============================================================================
// High-Level API
// =============================================================================

/**
 * Create 128-bit message with RS parity
 * @param {Float32Array|number[]} dataBits - 96 bits of data (Sync 16 + Time 32 + Auth 48)
 * @returns {Float32Array} - 128-bit message with RS parity
 */
export function createMessageWithRS(dataBits) {
  const dataBytes = dataBitsToBytes(dataBits);
  const codeword = rsEncode(dataBytes);
  return bytesToBits(codeword);
}

/**
 * Verify and correct message using RS
 * @param {Float32Array|number[]} message - 128-bit received message
 * @returns {{ isValid: boolean, corrected: Float32Array, errorsCorrected: number, confidence: number }}
 */
export function verifyRS(message) {
  const receivedBytes = bitsToBytes(message);
  const result = rsDecode(receivedBytes);

  // Calculate average bit confidence
  const confidences = Array.from(message).map(b => Math.abs(b - 0.5) * 2);
  const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;

  if (result.success) {
    // Re-encode to get the fully corrected codeword
    const correctedCodeword = rsEncode(result.data);
    const correctedBits = bytesToBits(correctedCodeword);

    return {
      isValid: true,
      corrected: correctedBits,
      errorsCorrected: result.corrected,
      confidence: avgConfidence
    };
  }

  return {
    isValid: false,
    corrected: new Float32Array(message),
    errorsCorrected: 0,
    confidence: avgConfidence
  };
}

/**
 * Extract data fields from message
 */
export function extractDataFields(message) {
  return {
    sync: Array.from(message.slice(0, 16)).map(b => b > 0.5 ? 1 : 0),
    timestamp: Array.from(message.slice(16, 48)).map(b => b > 0.5 ? 1 : 0),
    auth: Array.from(message.slice(48, 96)).map(b => b > 0.5 ? 1 : 0)
  };
}

/**
 * Compare two messages
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
    accuracy: matching / len,
    errorBits: len - matching
  };
}

/**
 * Format message as hex string
 */
export function messageToHex(message) {
  const bytes = bitsToBytes(message);
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Find low-confidence bits (useful for debugging)
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

export default {
  // Core RS
  rsEncode,
  rsDecode,

  // Bit conversion
  bitsToBytes,
  bytesToBits,
  dataBitsToBytes,
  dataBytesToBits,

  // High-level API
  createMessageWithRS,
  verifyRS,
  extractDataFields,
  compareMessages,
  messageToHex,
  findSuspiciousBits,

  // Constants
  NDATA,      // 12 bytes = 96 bits
  NSYM,       // 4 bytes = 32 bits parity
  NCODEWORD   // 16 bytes = 128 bits total
};
