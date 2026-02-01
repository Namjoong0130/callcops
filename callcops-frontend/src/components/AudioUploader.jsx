/**
 * AudioUploader Component
 * 
 * File upload interface with drag & drop support.
 */

import { useState, useRef } from 'react';

export function AudioUploader({ onFileSelect, disabled = false }) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState(null);
  const fileInputRef = useRef(null);
  
  const acceptedFormats = '.wav,.mp3,.ogg,.m4a,.flac';
  
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };
  
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (disabled) return;
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };
  
  const handleFileInput = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };
  
  const handleFile = (file) => {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/m4a', 'audio/flac', 'audio/x-wav'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|m4a|flac)$/i)) {
      alert('Please upload an audio file (WAV, MP3, OGG, M4A, or FLAC)');
      return;
    }
    
    setFileName(file.name);
    if (onFileSelect) {
      onFileSelect(file);
    }
  };
  
  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };
  
  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptedFormats}
        onChange={handleFileInput}
        className="hidden"
        disabled={disabled}
      />
      
      <div
        onClick={handleClick}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-xl p-6
          flex flex-col items-center justify-center
          cursor-pointer transition-all duration-200
          ${isDragging 
            ? 'border-primary bg-primary/10' 
            : 'border-gray-600 hover:border-primary/50 hover:bg-card/50'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        {/* Icon */}
        <svg
          className={`w-10 h-10 mb-3 ${isDragging ? 'text-primary' : 'text-gray-500'}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
        
        {/* Text */}
        <p className="text-sm font-medium text-gray-300 mb-1">
          {fileName ? fileName : 'Drop audio file here'}
        </p>
        <p className="text-xs text-gray-500">
          or click to browse
        </p>
        
        {/* Formats */}
        <div className="mt-3 flex flex-wrap gap-1 justify-center">
          {['WAV', 'MP3', 'OGG', 'M4A', 'FLAC'].map(format => (
            <span
              key={format}
              className="px-2 py-0.5 text-xs bg-gray-700/50 text-gray-400 rounded"
            >
              {format}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AudioUploader;
