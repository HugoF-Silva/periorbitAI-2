import React, { useRef, useState, useEffect } from 'react';
import { View, TouchableOpacity, StyleSheet, Text } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import type { Photo } from './types';

type CameraScreenProps = {
  navigation: any;
  route: {
    params: {
      setPhotos: React.Dispatch<React.SetStateAction<Photo[]>>;
      photos: Photo[];
    };
  };
};

export default function CameraScreen({
  navigation,
  route,
}: CameraScreenProps) {
  // useCameraPermissions comes from the legacy Camera
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<'front' | 'back'>('front');
  const cameraRef = useRef<any>(null);

  useEffect(() => {
    // if permissions have loaded and aren’t granted, ask
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, [permission]);

  if (!permission) return <View />;
  if (!permission.granted) {
    // permission denied
    return (
      <View style={styles.centered}>
        <Text>No access to camera</Text>
      </View>
    );
  }

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      const newPhoto: Photo = {
        uri: photo.uri,
        label: 'Just now',
      };
      const { setPhotos, photos } = route.params;
      setPhotos([newPhoto, ...photos]);
      navigation.goBack();
    }
  };

   return (
     <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing={facing}
        ref={cameraRef}
        ratio="16:9"
      >
         <View style={styles.camOverlay}>
           <TouchableOpacity
             style={styles.closeBtn}
             onPress={() => navigation.goBack()}
           >
             <Ionicons name="arrow-back" size={28} color="#fff" />
           </TouchableOpacity>
           <TouchableOpacity
             style={styles.snapBtn}
             onPress={takePicture}
           >
             <Ionicons name="camera" size={38} color="#fff" />
           </TouchableOpacity>
         </View>
      </CameraView>
     </View>
   );
 }

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  camOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    marginBottom: 24,
  },
  snapBtn: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#fff2',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  closeBtn: {
    position: 'absolute',
    top: 30,
    left: 20,
    zIndex: 10,
    backgroundColor: '#0008',
    borderRadius: 16,
    padding: 6,
  },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});
