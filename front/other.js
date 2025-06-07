import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

const MobileVisionApp = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [detections, setDetections] = useState([]);
  const animationRef = useRef(null);

  // COCO-SSD class names for common objects
  const classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
  ];

  useEffect(() => {
    // Load COCO-SSD model
    const loadModel = async () => {
      try {
        await tf.ready();
        const loadedModel = await tf.loadGraphModel(
          'https://tfhub.dev/tensorflow/tfjs-model/coco-ssd/1/default/1',
          { fromTFHub: true }
        );
        setModel(loadedModel);
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load model:', error);
        setIsLoading(false);
      }
    };

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
          };
        }
      } catch (error) {
        console.error('Camera access denied:', error);
      }
    };

    loadModel();
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
    if (model && videoRef.current && !isLoading) {
      detectFrame();
    }
  }, [model, isLoading]);

  const detectFrame = async () => {
    if (!model || !videoRef.current || videoRef.current.readyState !== 4) {
      animationRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    try {
      // Prepare input tensor
      const input = tf.browser.fromPixels(video);
      const normalized = input.div(255.0);
      const batched = normalized.expandDims(0);

      // Run detection
      const predictions = await model.executeAsync(batched);
      
      // Process predictions
      const [boxes, scores, classes, numDetections] = predictions;
      const boxesData = await boxes.data();
      const scoresData = await scores.data();
      const classesData = await classes.data();
      const numDetectionsData = await numDetections.data();

      const detectionsList = [];
      const numBoxes = numDetectionsData[0];
      
      for (let i = 0; i < numBoxes; i++) {
        const score = scoresData[i];
        if (score > 0.5) { // Confidence threshold
          const bbox = [
            boxesData[i * 4] * canvas.height,     // y1
            boxesData[i * 4 + 1] * canvas.width,  // x1
            boxesData[i * 4 + 2] * canvas.height, // y2
            boxesData[i * 4 + 3] * canvas.width   // x2
          ];
          
          detectionsList.push({
            bbox,
            class: classNames[classesData[i] - 1] || 'unknown',
            score: score
          });
        }
      }

      setDetections(detectionsList);

      // Clean up tensors
      input.dispose();
      normalized.dispose();
      batched.dispose();
      predictions.forEach(p => p.dispose());

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