import SwiftUI

struct PhotoCard: View {
    let photo: Photo
    let onTap: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Top: Photo preview (upper half, clipped)
            AsyncImage(url: URL(string: photo.url)) { image in
                image
                    .resizable()
                    .scaledToFill()
                    .frame(height: 100)
                    .clipped()
            } placeholder: {
                Color.gray.frame(height: 100)
            }
            // Bottom: Text label
            Text(photo.label)
                .font(.caption)
                .frame(maxWidth: .infinity)
                .padding(8)
                .background(Color.white.opacity(0.8))
        }
        .frame(width: 220, height: 140)
        .background(Color(.systemGray6))
        .cornerRadius(20)
        .shadow(radius: 4)
        .onTapGesture { onTap() }
    }
}
