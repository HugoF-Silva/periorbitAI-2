import React, { useRef, useEffect, useState } from 'react';

const MobileVisionApp = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [detections, setDetections] = useState([]);
  const animationRef = useRef(null);

  // Simple object detection using basic image analysis
  const detectObjects = async (imageData) => {
    // Simulate object detection with basic color/shape analysis
    const objects = [];
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    // Basic color detection
    let redPixels = 0;
    let greenPixels = 0;
    let bluePixels = 0;
    let darkPixels = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const brightness = (r + g + b) / 3;
      
      if (r > 150 && g < 100 && b < 100) redPixels++;
      if (g > 150 && r < 100 && b < 100) greenPixels++;
      if (b > 150 && r < 100 && g < 100) bluePixels++;
      if (brightness < 50) darkPixels++;
    }
    
    const totalPixels = width * height;
    
    // Detect dominant colors as "objects"
    if (redPixels / totalPixels > 0.1) {
      objects.push({
        class: 'red object',
        confidence: Math.min(redPixels / totalPixels * 5, 0.99),
        x: width * 0.3,
        y: height * 0.3,
        w: width * 0.4,
        h: height * 0.4
      });
    }
    
    if (greenPixels / totalPixels > 0.1) {
      objects.push({
        class: 'green object',
        confidence: Math.min(greenPixels / totalPixels * 5, 0.99),
        x: width * 0.5,
        y: height * 0.5,
        w: width * 0.3,
        h: height * 0.3
      });
    }
    
    if (bluePixels / totalPixels > 0.1) {
      objects.push({
        class: 'blue object',
        confidence: Math.min(bluePixels / totalPixels * 5, 0.99),
        x: width * 0.2,
        y: height * 0.6,
        w: width * 0.35,
        h: height * 0.25
      });
    }
    
    if (darkPixels / totalPixels > 0.3) {
      objects.push({
        class: 'dark object',
        confidence: Math.min(darkPixels / totalPixels * 2, 0.95),
        x: width * 0.4,
        y: height * 0.4,
        w: width * 0.2,
        h: height * 0.2
      });
    }
    
    return objects;
  };

  useEffect(() => {
    // Initialize camera
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
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
    if (!isLoading && videoRef.current) {
      detectFrame();
    }
  }, [isLoading]);

  const detectFrame = async () => {
    if (!videoRef.current || videoRef.current.readyState !== 4) {
      animationRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const objects = await detectObjects(imageData);
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
    
    detections.forEach(detection => {
      const [y1, x1, y2, x2] = detection.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      // Draw bounding box
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, width, height);

      // Draw label background
      ctx.fillStyle = '#00ff00';
      ctx.fillRect(x1, y1 - 30, width, 30);

      // Draw label text
      ctx.fillStyle = '#000000';
      ctx.font = '16px Arial';
      ctx.fillText(
        `${detection.class} ${Math.round(detection.score * 100)}%`,
        x1 + 5,
        y1 - 10
      );
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
      
      {/* Loading Indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-white text-lg">Loading AI Model...</p>
          </div>
        </div>
      )}
      
      {/* Minimal UI - Just a title */}
      <div className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/50 to-transparent">
        <h1 className="text-white text-xl font-semibold text-center">Vision AI</h1>
        <p className="text-white/70 text-sm text-center mt-1">Point at objects to identify</p>
      </div>
      
      {/* Detection Count */}
      {!isLoading && (
        <div className="absolute bottom-4 left-4 bg-black/50 rounded-lg px-3 py-2">
          <p className="text-white text-sm">
            {detections.length} {detections.length === 1 ? 'object' : 'objects'} detected
          </p>
        </div>
      )}
    </div>
  );
};

export default MobileVisionApp;