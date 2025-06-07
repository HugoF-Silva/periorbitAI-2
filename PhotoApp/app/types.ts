export type Photo = {
  uri: string;
  label: string;
};

export type RootStackParamList = {
  Main: undefined;                        // no params
  Camera: {
    setPhotos: React.Dispatch<React.SetStateAction<Photo[]>>;
    photos: Photo[];
  };
};
