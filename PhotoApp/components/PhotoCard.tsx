import React from 'react';
import { TouchableOpacity, View, Image, Text, StyleSheet, Dimensions } from 'react-native';

// Define the props type
type Photo = {
  uri: string;
  label: string;
};

type PhotoCardProps = {
  photo: Photo;
  onPress: () => void;
};

const CARD_WIDTH = Dimensions.get('window').width * 0.9;
const CARD_HEIGHT = 140;

export default function PhotoCard({ photo, onPress }: PhotoCardProps) {
  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.8}>
      <View style={styles.imageContainer}>
        <Image source={{ uri: photo.uri }} style={styles.image} />
      </View>
      <View style={styles.textContainer}>
        <Text style={styles.label}>{photo.label}</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    borderRadius: 20,
    backgroundColor: '#f6f7fb',
    marginVertical: 8,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    overflow: 'hidden',
    alignSelf: 'center',
  },
  imageContainer: {
    flex: 2,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    resizeMode: 'cover',
  },
  textContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  label: {
    fontSize: 16,
    color: '#333',
  },
});
