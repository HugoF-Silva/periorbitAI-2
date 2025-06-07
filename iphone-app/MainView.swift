import SwiftUI

struct Photo: Identifiable, Decodable {
    let id: String
    let url: String
    let label: String
}

struct MainView: View {
    @State private var photos: [Photo] = []
    @State private var showCamera = false
    @State private var selectedPhoto: Photo?
    @State private var showPhotoFullScreen = false

    var body: some View {
        NavigationView {
            ZStack(alignment: .topTrailing) {
                ScrollView {
                    if photos.isEmpty {
                        VStack { Spacer(); Text("No photos yet").foregroundColor(.secondary); Spacer() }
                    } else {
                        VStack(spacing: 16) {
                            ForEach(photos) { photo in
                                PhotoCard(photo: photo) {
                                    selectedPhoto = photo
                                    showPhotoFullScreen = true
                                }
                            }
                        }
                        .padding()
                    }
                }
                // Camera button (upper right)
                Button(action: { showCamera = true }) {
                    Image(systemName: "camera")
                        .font(.title2)
                        .padding(16)
                        .background(.thinMaterial)
                        .clipShape(Circle())
                        .shadow(radius: 2)
                }
                .padding(.top, 40)
                .padding(.trailing, 20)
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $showCamera) {
                CameraView { newPhoto in
                    // Optionally upload to backend here, then refresh
                    photos.append(newPhoto)
                }
            }
            .fullScreenCover(item: $selectedPhoto) { photo in
                VStack {
                    AsyncImage(url: URL(string: photo.url)) { image in
                        image
                            .resizable()
                            .scaledToFit()
                    } placeholder: {
                        Color.gray
                    }
                    Button("Close") { selectedPhoto = nil }
                        .padding()
                }
                .background(Color.black.opacity(0.95))
            }
            .onAppear(perform: loadPhotos)
        }
    }

    func loadPhotos() {
        // Replace with your API endpoint
        guard let url = URL(string: "https://your-backend/photos") else { return }
        URLSession.shared.dataTask(with: url) { data, _, _ in
            if let data = data, let fetched = try? JSONDecoder().decode([Photo].self, from: data) {
                DispatchQueue.main.async { photos = fetched }
            }
        }.resume()
    }
}
