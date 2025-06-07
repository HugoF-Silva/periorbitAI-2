// screens/MainScreen.tsx
import React, { useState } from 'react';
import {
  View,
  ScrollView,
  Modal,
  Image,
  Text,
  TouchableOpacity,
  StyleSheet
} from 'react-native';
import PhotoCard from '../components/PhotoCard';
import { Ionicons } from '@expo/vector-icons';
import { Photo } from './types';


export default function MainScreen({ navigation }: any) {
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

  return (
    <View style={styles.container}>
      {/* Camera Button */}
      <TouchableOpacity
        style={styles.cameraButton}
        onPress={() => navigation.navigate('Camera', { setPhotos, photos })}
      >
        <Ionicons name="camera" size={28} color="#0066cc" />
      </TouchableOpacity>

      {/* Photo List */}
      <ScrollView contentContainerStyle={{ paddingTop: 70, paddingBottom: 40 }}>
        {photos.length === 0 ? (
          <View style={styles.noPhotoContainer}>
            <Text style={styles.noPhotoText}>No photos yet</Text>
          </View>
        ) : (
          photos.map((photo, idx) => (
            <PhotoCard
              key={idx}
              photo={photo}
              onPress={() => setSelectedPhoto(photo)}
            />
          ))
        )}
      </ScrollView>

      {/* Fullscreen Modal */}
      <Modal visible={!!selectedPhoto} transparent animationType="fade">
        <View style={styles.modalBg}>
          {selectedPhoto && (
            <Image
              source={{ uri: selectedPhoto.uri }}
              style={styles.fullImage}
            />
          )}
          <TouchableOpacity
            style={styles.closeButton}
            onPress={() => setSelectedPhoto(null)}
          >
            <Ionicons name="close" size={32} color="#fff" />
          </TouchableOpacity>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  cameraButton: {
    position: 'absolute',
    top: 40,
    right: 24,
    zIndex: 10,
    backgroundColor: '#fff',
    borderRadius: 24,
    padding: 8,
    elevation: 2,
    shadowColor: '#000',
    shadowOpacity: 0.09,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
  },
  noPhotoContainer: {
    flex: 1,
    alignItems: 'center',
    marginTop: 60,
  },
  noPhotoText: { color: '#888', fontSize: 18 },
  modalBg: {
    flex: 1,
    backgroundColor: '#000a',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImage: {
    width: '90%',
    height: '70%',
    borderRadius: 20,
    resizeMode: 'contain',
    marginTop: 40,
  },
  closeButton: {
    position: 'absolute',
    top: 60,
    right: 30,
    padding: 10,
  },
});
