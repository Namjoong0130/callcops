/**
 * Test RS with Interleaving
 */

import { createMessageWithRS, verifyRS, bitsToBytes, bytesToBits } from './reedSolomon.js';

// Create a test message (96 bits data)
const dataBits = new Float32Array(96);
// Sync pattern: 1010...
for (let i = 0; i < 16; i++) dataBits[i] = i % 2;
// Timestamp: all 1s
for (let i = 16; i < 48; i++) dataBits[i] = 1;
// Auth: alternating
for (let i = 48; i < 96; i++) dataBits[i] = i % 2;

console.log('=== Test RS with Bit Interleaving ===\n');

// Create message with RS
const encoded = createMessageWithRS(dataBits);
console.log('Original 128-bit message created');

// Test 1: No errors
console.log('\n--- Test 1: No errors ---');
let result = verifyRS(encoded);
console.log(`Valid: ${result.isValid}, Errors corrected: ${result.errorsCorrected}`);

// Test 2: 5 consecutive bit errors (should work with interleaving!)
console.log('\n--- Test 2: 5 consecutive bit errors (bits 0-4) ---');
const corrupted1 = new Float32Array(encoded);
for (let i = 0; i < 5; i++) {
  corrupted1[i] = 1 - corrupted1[i]; // Flip bits 0,1,2,3,4
}
result = verifyRS(corrupted1);
console.log(`Valid: ${result.isValid}, Errors corrected: ${result.errorsCorrected} byte(s)`);
if (result.isValid) {
  // Check if corrected matches original
  let matches = 0;
  for (let i = 0; i < 128; i++) {
    if ((result.corrected[i] > 0.5) === (encoded[i] > 0.5)) matches++;
  }
  console.log(`Accuracy after correction: ${(matches/128*100).toFixed(1)}%`);
}

// Test 3: 8 consecutive bit errors
console.log('\n--- Test 3: 8 consecutive bit errors (bits 0-7) ---');
const corrupted2 = new Float32Array(encoded);
for (let i = 0; i < 8; i++) {
  corrupted2[i] = 1 - corrupted2[i];
}
result = verifyRS(corrupted2);
console.log(`Valid: ${result.isValid}, Errors corrected: ${result.errorsCorrected} byte(s)`);
if (result.isValid) {
  let matches = 0;
  for (let i = 0; i < 128; i++) {
    if ((result.corrected[i] > 0.5) === (encoded[i] > 0.5)) matches++;
  }
  console.log(`Accuracy after correction: ${(matches/128*100).toFixed(1)}%`);
}

// Test 4: 16 consecutive bit errors (max for 2-byte correction with interleaving)
console.log('\n--- Test 4: 16 consecutive bit errors (bits 0-15) ---');
const corrupted3 = new Float32Array(encoded);
for (let i = 0; i < 16; i++) {
  corrupted3[i] = 1 - corrupted3[i];
}
result = verifyRS(corrupted3);
console.log(`Valid: ${result.isValid}, Errors corrected: ${result.errorsCorrected} byte(s)`);
if (result.isValid) {
  let matches = 0;
  for (let i = 0; i < 128; i++) {
    if ((result.corrected[i] > 0.5) === (encoded[i] > 0.5)) matches++;
  }
  console.log(`Accuracy after correction: ${(matches/128*100).toFixed(1)}%`);
}

// Test 5: 5 scattered bit errors (one every 25 bits)
console.log('\n--- Test 5: 5 scattered bit errors (bits 0, 25, 50, 75, 100) ---');
const corrupted4 = new Float32Array(encoded);
[0, 25, 50, 75, 100].forEach(i => {
  corrupted4[i] = 1 - corrupted4[i];
});
result = verifyRS(corrupted4);
console.log(`Valid: ${result.isValid}, Errors corrected: ${result.errorsCorrected} byte(s)`);

// Test 6: Roundtrip test
console.log('\n--- Test 6: Roundtrip bits → bytes → bits ---');
const testBits = new Float32Array(128);
for (let i = 0; i < 128; i++) testBits[i] = Math.random() > 0.5 ? 1 : 0;
const bytes = bitsToBytes(testBits);
const recovered = bytesToBits(bytes);
let roundtripMatch = 0;
for (let i = 0; i < 128; i++) {
  if ((testBits[i] > 0.5) === (recovered[i] > 0.5)) roundtripMatch++;
}
console.log(`Roundtrip accuracy: ${(roundtripMatch/128*100).toFixed(1)}%`);

console.log('\n=== Tests Complete ===');
