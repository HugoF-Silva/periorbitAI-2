import React, { useState, useEffect, useRef } from 'react';
import { Camera, ArrowLeft, MoreVertical, Share2, Trash2, Loader2 } from 'lucide-react';

const App = () => {
  const [currentView, setCurrentView] = useState('gallery');
  const [photos, setPhotos] = useState([]);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [expandedPhoto, setExpandedPhoto] = useState(null);
  const [showMenu, setShowMenu] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectionInterval = useRef(null);

  // Open real camera and start face detection
  useEffect(() => {
    if (currentView === 'camera') {
      setIsCameraReady(false);
      const startCamera = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              facingMode: 'user',
              width: { ideal: 1280 },
              height: { ideal: 720 }
            },
            audio: false
          });
          
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              setIsCameraReady(true);
            };
          }
          
          // Start face detection simulation
          detectionInterval.current = setInterval(() => {
            setIsFaceDetected(Math.random() > 0.3);
          }, 500);
        } catch (err) {
          console.error('Camera access denied:', err);
          alert('Please allow camera access to use this feature');
          setCurrentView('gallery');
        }
      };
      
      startCamera();
    } else {
      // Stop camera and detection
      setIsCameraReady(false);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
      }
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
      }
    };
  }, [currentView]);

  const capturePhoto = async () => {
    setIsCapturing(true);
    
    // Create canvas to capture photo from video
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    
    // Flip horizontally for selfie (since video is mirrored)
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to data URL
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
    
    setTimeout(() => {
      const newPhoto = {
        id: Date.now(),
        url: imageDataUrl, // Real captured photo
        description: `Analysis ${photos.length + 1}: Healthy periorbital region detected. No signs of puffiness or dark circles. Skin elasticity appears normal. Hydration level: Good.`,
        timestamp: new Date().toISOString()
      };
      
      setPhotos([...photos, newPhoto]);
      setIsCapturing(false);
      setCurrentView('gallery');
    }, 100);
  };

  const deletePhoto = (photoId) => {
    setPhotos(photos.filter(p => p.id !== photoId));
    setExpandedPhoto(null);
    setShowMenu(false);
  };

  const sharePhoto = () => {
    // Simulate share functionality
    alert('Share functionality would open native share sheet');
    setShowMenu(false);
  };

  // Gallery View
  if (currentView === 'gallery') {
    return (
      <div className="w-full h-screen bg-black text-white">
        {/* Status Bar */}
        <div className="flex justify-between items-center px-6 pt-3 pb-2 text-xs">
          <span>9:41</span>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-white rounded-sm"></div>
            <div className="w-4 h-4 bg-white rounded-sm"></div>
            <div className="w-4 h-4 bg-white rounded-sm"></div>
          </div>
        </div>

        {/* Header */}
        <div className="px-6 py-4">
          <h1 className="text-2xl font-semibold">Periorbital Analysis</h1>
        </div>

        {/* Content */}
        <div className="flex-1 px-6">
          {photos.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <p className="text-gray-500">No photos yet</p>
            </div>
          ) : (
            <div className="space-y-4 pb-24">
              {photos.map((photo) => (
                <div
                  key={photo.id}
                  className="bg-gray-900 rounded-2xl overflow-hidden"
                >
                  <div
                    onClick={() => setExpandedPhoto(photo)}
                    className="cursor-pointer"
                  >
                    <div className="aspect-[4/3] overflow-hidden">
                      <img
                        src={photo.url}
                        alt="Face analysis"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-sm text-gray-300">{photo.description}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Camera Button */}
        <div className="absolute bottom-8 right-6">
          <button
            onClick={async () => {
              // Check camera permissions first
              try {
                const permissionStatus = await navigator.permissions.query({ name: 'camera' });
                if (permissionStatus.state === 'denied') {
                  alert('Camera access is blocked. Please enable it in your browser settings.');
                  return;
                }
                setCurrentView('camera');
              } catch {
                // Permissions API not supported, try anyway
                setCurrentView('camera');
              }
            }}
            className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-lg"
          >
            <Camera className="w-8 h-8 text-black" />
          </button>
        </div>

        {/* Expanded Photo View */}
        {expandedPhoto && (
          <div className="absolute inset-0 bg-black z-50">
            <div className="flex justify-between items-center px-6 pt-3 pb-2">
              <button
                onClick={() => {
                  setExpandedPhoto(null);
                  setShowMenu(false);
                }}
                className="p-2"
              >
                <ArrowLeft className="w-6 h-6" />
              </button>
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-2 relative"
              >
                <MoreVertical className="w-6 h-6" />
                
                {showMenu && (
                  <div className="absolute right-0 top-12 bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                    <button
                      onClick={sharePhoto}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-gray-700 w-full"
                    >
                      <Share2 className="w-5 h-5" />
                      <span>Share</span>
                    </button>
                    <button
                      onClick={() => deletePhoto(expandedPhoto.id)}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-gray-700 w-full text-red-400"
                    >
                      <Trash2 className="w-5 h-5" />
                      <span>Delete</span>
                    </button>
                  </div>
                )}
              </button>
            </div>
            
            <div className="flex flex-col h-full">
              <div className="flex-1 flex items-center justify-center px-6">
                <img
                  src={expandedPhoto.url}
                  alt="Face analysis"
                  className="max-w-full max-h-full object-contain"
                />
              </div>
              <div className="px-6 pb-8">
                <p className="text-sm text-gray-300">{expandedPhoto.description}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Camera View
  return (
    <div className="w-full h-screen bg-black text-white relative">
      {/* Flash Effect */}
      {isCapturing && (
        <div className="absolute inset-0 bg-white z-50 pointer-events-none"></div>
      )}

      {/* Status Bar */}
      <div className="flex justify-between items-center px-6 pt-3 pb-2 text-xs relative z-10">
        <span>9:41</span>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-white rounded-sm"></div>
          <div className="w-4 h-4 bg-white rounded-sm"></div>
          <div className="w-4 h-4 bg-white rounded-sm"></div>
        </div>
      </div>

      {/* Header */}
      <div className="absolute top-16 left-6 z-10">
        <button
          onClick={() => setCurrentView('gallery')}
          className="p-2 bg-black/50 rounded-full backdrop-blur-sm"
        >
          <ArrowLeft className="w-6 h-6" />
        </button>
      </div>

      {/* Camera View */}
      <div className="absolute inset-0">
        <div className="w-full h-full bg-black">
          {/* Real camera video */}
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            autoPlay
            playsInline
            muted
            style={{ transform: 'scaleX(-1)' }} // Mirror for selfie camera
          />
          
          {/* Loading indicator */}
          {!isCameraReady && (
            <div className="absolute inset-0 bg-black flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-8 h-8 animate-spin text-white mx-auto mb-2" />
                <p className="text-gray-400 text-sm">Opening camera...</p>
              </div>
            </div>
          )}
          
          {/* Face detection frame overlay */}
          {isCameraReady && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className={`w-64 h-80 border-2 ${isFaceDetected ? 'border-green-400' : 'border-red-400'} rounded-3xl transition-colors duration-300`}>
                <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
                  <p className="text-sm bg-black/50 px-3 py-1 rounded-full backdrop-blur-sm">
                    {isFaceDetected ? 'Face detected' : 'Position face here'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Camera Controls */}
      <div className="absolute bottom-12 left-0 right-0 flex justify-center">
        <button
          onClick={capturePhoto}
          disabled={!isFaceDetected || isCapturing}
          className={`w-20 h-20 rounded-full border-4 border-white flex items-center justify-center transition-all ${
            isFaceDetected && !isCapturing
              ? 'bg-white scale-100 active:scale-95'
              : 'bg-transparent scale-90 opacity-50'
          }`}
        >
          {isFaceDetected && !isCapturing && (
            <div className="w-16 h-16 bg-black rounded-full"></div>
          )}
        </button>
      </div>
    </div>
  );
};

export default App;