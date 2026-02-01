/**
 * Audio Comparison Utilities
 * 
 * Calculate similarity metrics between original and watermarked audio.
 */

/**
 * Calculate Signal-to-Noise Ratio (SNR) in dB
 * @param {Float32Array} original - Original audio samples
 * @param {Float32Array} watermarked - Watermarked audio samples
 * @returns {number} SNR in dB
 */
export function calculateSNR(original, watermarked) {
  const len = Math.min(original.length, watermarked.length);
  
  let signalPower = 0;
  let noisePower = 0;
  
  for (let i = 0; i < len; i++) {
    signalPower += original[i] * original[i];
    const noise = original[i] - watermarked[i];
    noisePower += noise * noise;
  }
  
  signalPower /= len;
  noisePower /= len;
  
  if (noisePower === 0) return Infinity;
  if (signalPower === 0) return 0;
  
  return 10 * Math.log10(signalPower / noisePower);
}

/**
 * Calculate Peak Signal-to-Noise Ratio (PSNR) in dB
 * @param {Float32Array} original 
 * @param {Float32Array} watermarked 
 * @returns {number} PSNR in dB
 */
export function calculatePSNR(original, watermarked) {
  const len = Math.min(original.length, watermarked.length);
  
  let mse = 0;
  for (let i = 0; i < len; i++) {
    const diff = original[i] - watermarked[i];
    mse += diff * diff;
  }
  mse /= len;
  
  if (mse === 0) return Infinity;
  
  // Peak value for normalized audio is 1.0
  return 10 * Math.log10(1.0 / mse);
}

/**
 * Calculate Root Mean Square Error
 * @param {Float32Array} original 
 * @param {Float32Array} watermarked 
 * @returns {number} RMSE
 */
export function calculateRMSE(original, watermarked) {
  const len = Math.min(original.length, watermarked.length);
  
  let mse = 0;
  for (let i = 0; i < len; i++) {
    const diff = original[i] - watermarked[i];
    mse += diff * diff;
  }
  mse /= len;
  
  return Math.sqrt(mse);
}

/**
 * Calculate correlation coefficient
 * @param {Float32Array} original 
 * @param {Float32Array} watermarked 
 * @returns {number} Correlation coefficient (-1 to 1)
 */
export function calculateCorrelation(original, watermarked) {
  const len = Math.min(original.length, watermarked.length);
  
  // Calculate means
  let meanOrig = 0, meanWater = 0;
  for (let i = 0; i < len; i++) {
    meanOrig += original[i];
    meanWater += watermarked[i];
  }
  meanOrig /= len;
  meanWater /= len;
  
  // Calculate correlation
  let num = 0, denomOrig = 0, denomWater = 0;
  for (let i = 0; i < len; i++) {
    const diffOrig = original[i] - meanOrig;
    const diffWater = watermarked[i] - meanWater;
    num += diffOrig * diffWater;
    denomOrig += diffOrig * diffOrig;
    denomWater += diffWater * diffWater;
  }
  
  const denom = Math.sqrt(denomOrig * denomWater);
  if (denom === 0) return 1;
  
  return num / denom;
}

/**
 * Calculate all audio comparison metrics
 * @param {Float32Array} original 
 * @param {Float32Array} watermarked 
 * @returns {object} All metrics
 */
export function calculateAllMetrics(original, watermarked) {
  return {
    snr: calculateSNR(original, watermarked),
    psnr: calculatePSNR(original, watermarked),
    rmse: calculateRMSE(original, watermarked),
    correlation: calculateCorrelation(original, watermarked),
    durationOriginal: original.length / 8000,
    durationWatermarked: watermarked.length / 8000
  };
}

/**
 * Get quality rating based on SNR
 * @param {number} snr - SNR in dB
 * @returns {{ rating: string, color: string, description: string }}
 */
export function getQualityRating(snr) {
  if (snr >= 40) {
    return { rating: 'Excellent', color: 'green', description: 'Imperceptible watermark' };
  } else if (snr >= 30) {
    return { rating: 'Good', color: 'lime', description: 'Barely audible' };
  } else if (snr >= 20) {
    return { rating: 'Fair', color: 'yellow', description: 'Slightly audible' };
  } else if (snr >= 10) {
    return { rating: 'Poor', color: 'orange', description: 'Noticeably audible' };
  } else {
    return { rating: 'Bad', color: 'red', description: 'Very audible noise' };
  }
}

export default {
  calculateSNR,
  calculatePSNR,
  calculateRMSE,
  calculateCorrelation,
  calculateAllMetrics,
  getQualityRating
};
