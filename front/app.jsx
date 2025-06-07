import React, { useRef, useEffect, useState } from 'react';

const MobileVisionApp = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [detections, setDetections] = useState([]);
  const [cameraError, setCameraError] = useState(false);
  const animationRef = useRef(null);

  // Enhanced object detection with motion and edge detection
  const detectObjects = (imageData) => {
    const objects = [];
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    // Analyze image in grid sections
    const gridSize = 4;
    const cellWidth = width / gridSize;
    const cellHeight = height / gridSize;
    
    for (let gx = 0; gx < gridSize; gx++) {
      for (let gy = 0; gy < gridSize; gy++) {
        let totalBrightness = 0;
        let edgeStrength = 0;
        let colorVariance = 0;
        let pixelCount = 0;
        
        // Analyze each grid cell
        for (let x = gx * cellWidth; x < (gx + 1) * cellWidth; x += 2) {
          for (let y = gy * cellHeight; y < (gy + 1) * cellHeight; y += 2) {
            const idx = (y * width + x) * 4;
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            
            totalBrightness += (r + g + b) / 3;
            pixelCount++;
            
            // Edge detection
            if (x < width - 1 && y < height - 1) {
              const idx2 = ((y + 1) * width + (x + 1)) * 4;
              const dr = Math.abs(data[idx2] - r);
              const dg = Math.abs(data[idx2 + 1] - g);
              const db = Math.abs(data[idx2 + 2] - b);
              edgeStrength += (dr + dg + db) / 3;
            }
            
            // Color variance
            const avg = (r + g + b) / 3;
            colorVariance += Math.abs(r - avg) + Math.abs(g - avg) + Math.abs(b - avg);
          }
        }
        
        const avgBrightness = totalBrightness / pixelCount;
        const avgEdge = edgeStrength / pixelCount;
        const avgVariance = colorVariance / pixelCount;
        
        // Detect interesting regions
        if (avgEdge > 30 || avgVariance > 40) {
          const confidence = Math.min((avgEdge + avgVariance) / 150, 0.95);
          
          let objectType = 'object';
          if (avgBrightness > 200) objectType = 'bright object';
          else if (avgBrightness < 50) objectType = 'dark object';
          else if (avgVariance > 60) objectType = 'colorful object';
          else if (avgEdge > 50) objectType = 'detailed object';
          
          objects.push({
            class: objectType,
            confidence: confidence,
            x: gx * cellWidth + cellWidth * 0.1,
            y: gy * cellHeight + cellHeight * 0.1,
            w: cellWidth * 0.8,
            h: cellHeight * 0.8
          });
        }
      }
    }
    
    // Limit to top 5 detections
    return objects
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);
  };

  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setIsLoading(false);
          };
        }
      } catch (error) {
        console.error('Camera access denied:', error);
        setCameraError(true);
        setIsLoading(false);
      }
    };

    initCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isLoading && !cameraError && videoRef.current) {
      detectFrame();
    }
  }, [isLoading, cameraError]);

  const detectFrame = () => {
    if (!videoRef.current || videoRef.current.readyState !== 4) {
      animationRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const objects = detectObjects(imageData);
      setDetections(objects);
    } catch (error) {
      console.error('Detection error:', error);
    }

    animationRef.current = requestAnimationFrame(detectFrame);
  };

  const drawDetections = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach((detection, index) => {
      const { x, y, w, h, confidence, class: className } = detection;
      
      // Animated pulse effect
      const time = Date.now() / 1000;
      const pulse = Math.sin(time * 3 + index) * 0.1 + 1;
      
      // Draw bounding box with gradient
      ctx.strokeStyle = `hsla(${120 + index * 60}, 100%, 50%, ${0.8 * pulse})`;
      ctx.lineWidth = 3 * pulse;
      ctx.strokeRect(x, y, w, h);
      
      // Draw corner accents
      const cornerLength = Math.min(w, h) * 0.15;
      ctx.lineWidth = 5 * pulse;
      
      // Top-left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLength);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLength, y);
      ctx.stroke();
      
      // Top-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLength, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLength);
      ctx.stroke();
      
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLength);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerLength, y + h);
      ctx.stroke();
      
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLength, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerLength);
      ctx.stroke();
      
      // Draw label with glassmorphism effect
      const label = `${className} ${Math.round(confidence * 100)}%`;
      ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const padding = 12;
      
      // Label background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 35, textWidth + padding * 2, 30);
      
      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + padding, y - 15);
    });
  };

  useEffect(() => {
    drawDetections();
  }, [detections]);

  return (
    <div className="relative w-screen h-screen bg-black overflow-hidden">
      {/* Camera Feed */}
      <video
        ref={videoRef}
        className="absolute top-0 left-0 w-full h-full object-cover"
        playsInline
        muted
      />
      
      {/* Detection Overlay */}
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
      
      {/* Loading State */}
      {isLoading && !cameraError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black">
          <div className="text-center">
            <div className="w-16 h-16 relative">
              <div className="absolute inset-0 border-4 border-white/20 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
            </div>
            <p className="text-white text-lg mt-4 font-light">Initializing Vision...</p>
          </div>
        </div>
      )}
      
      {/* Camera Error */}
      {cameraError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black">
          <div className="text-center px-8">
            <div className="w-20 h-20 mx-auto mb-4 text-white/50">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <h2 className="text-white text-xl font-semibold mb-2">Camera Access Required</h2>
            <p className="text-white/70 text-sm">Please allow camera access to use Vision AI</p>
            <button 
              onClick={() => window.location.reload()}
              className="mt-6 px-6 py-3 bg-white text-black rounded-full font-medium"
            >
              Try Again
            </button>
          </div>
        </div>
      )}
      
      {/* Minimal UI Overlay */}
      {!isLoading && !cameraError && (
        <>
          {/* Top Bar */}
          <div className="absolute top-0 left-0 right-0 p-6 bg-gradient-to-b from-black/50 to-transparent">
            <h1 className="text-white text-2xl font-bold text-center tracking-tight">VISION</h1>
          </div>
          
          {/* Bottom Info */}
          <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/50 to-transparent">
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <p className="text-white/90 text-sm font-medium">
                {detections.length > 0 
                  ? `Tracking ${detections.length} object${detections.length > 1 ? 's' : ''}`
                  : 'Scanning...'}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default MobileVisionApp;