import React, { useState, useEffect } from 'react';
import { Camera, Zap, X, Check, Loader2 } from 'lucide-react';

const App = () => {
  const [isLaunching, setIsLaunching] = useState(true);
  const [isScanning, setIsScanning] = useState(false);
  const [detectedObject, setDetectedObject] = useState(null);
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    // Simulate launch screen
    const timer = setTimeout(() => setIsLaunching(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  const startScanning = () => {
    setIsScanning(true);
    setDetectedObject(null);
    
    // Simulate object detection after 2 seconds
    setTimeout(() => {
      const objects = [
        { name: 'Coffee Cup', confidence: 0.94 },
        { name: 'Laptop', confidence: 0.89 },
        { name: 'Plant', confidence: 0.91 },
        { name: 'Book', confidence: 0.87 },
        { name: 'Phone', confidence: 0.92 }
      ];
      const detected = objects[Math.floor(Math.random() * objects.length)];
      setDetectedObject(detected.name);
      setConfidence(detected.confidence);
      setIsScanning(false);
    }, 2000);
  };

  const reset = () => {
    setDetectedObject(null);
    setConfidence(0);
  };

  // Launch Screen
  if (isLaunching) {
    return (
      <div className="w-full h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-24 h-24 bg-white rounded-3xl flex items-center justify-center mb-6 mx-auto">
            <Camera className="w-14 h-14 text-black" />
          </div>
          <h1 className="text-white text-2xl font-semibold mb-2">Vision</h1>
          <p className="text-gray-400 text-sm">See the world differently</p>
        </div>
      </div>
    );
  }

  // Main App
  return (
    <div className="w-full h-screen bg-black text-white overflow-hidden">
      {/* Status Bar */}
      <div className="flex justify-between items-center px-6 pt-3 pb-2 text-xs">
        <span>9:41</span>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-white rounded-sm"></div>
          <div className="w-4 h-4 bg-white rounded-sm"></div>
          <div className="w-4 h-4 bg-white rounded-sm"></div>
        </div>
      </div>

      {/* Camera View */}
      <div className="relative h-full">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 to-gray-800">
          {/* Simulated camera feed */}
          <div className="absolute inset-0 opacity-20">
            <div className="h-full w-full bg-gradient-to-b from-transparent via-white to-transparent animate-pulse"></div>
          </div>
          
          {/* Viewfinder */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="relative">
              <div className="w-64 h-64 border-2 border-white/30 rounded-3xl">
                {isScanning && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 animate-spin text-white" />
                  </div>
                )}
                
                {detectedObject && !isScanning && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 rounded-3xl backdrop-blur-sm">
                    <Check className="w-12 h-12 text-green-400 mb-3" />
                    <p className="text-xl font-semibold">{detectedObject}</p>
                    <p className="text-sm text-gray-300 mt-1">{(confidence * 100).toFixed(0)}% confidence</p>
                  </div>
                )}
              </div>
              
              {/* Corner markers */}
              <div className="absolute -top-2 -left-2 w-8 h-8 border-t-2 border-l-2 border-white rounded-tl-lg"></div>
              <div className="absolute -top-2 -right-2 w-8 h-8 border-t-2 border-r-2 border-white rounded-tr-lg"></div>
              <div className="absolute -bottom-2 -left-2 w-8 h-8 border-b-2 border-l-2 border-white rounded-bl-lg"></div>
              <div className="absolute -bottom-2 -right-2 w-8 h-8 border-b-2 border-r-2 border-white rounded-br-lg"></div>
            </div>
          </div>
        </div>

        {/* Top Controls */}
        <div className="absolute top-8 left-0 right-0 px-6">
          <div className="bg-black/30 backdrop-blur-md rounded-full px-4 py-2 mx-auto w-fit">
            <p className="text-sm text-center">
              {isScanning ? 'Analyzing...' : detectedObject ? 'Object detected' : 'Point at object'}
            </p>
          </div>
        </div>

        {/* Bottom Controls */}
        <div className="absolute bottom-0 left-0 right-0 pb-12">
          <div className="flex justify-center items-center gap-6">
            {/* Reset button */}
            {detectedObject && (
              <button
                onClick={reset}
                className="w-14 h-14 bg-white/20 backdrop-blur-md rounded-full flex items-center justify-center transition-all active:scale-95"
              >
                <X className="w-6 h-6" />
              </button>
            )}
            
            {/* Main scan button */}
            <button
              onClick={startScanning}
              disabled={isScanning}
              className="relative"
            >
              <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center transition-all active:scale-95 disabled:opacity-50">
                <Zap className="w-8 h-8 text-black" />
              </div>
              {isScanning && (
                <div className="absolute inset-0 rounded-full border-4 border-white/30 animate-ping"></div>
              )}
            </button>
          </div>
          
          {/* Instructions */}
          {!detectedObject && !isScanning && (
            <p className="text-center text-sm text-gray-400 mt-6">
              Tap to identify objects
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;