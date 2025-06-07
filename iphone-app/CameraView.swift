import SwiftUI
import UIKit

struct CameraView: UIViewControllerRepresentable {
    var onCapture: (Photo) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera
        picker.cameraDevice = .front
        picker.allowsEditing = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: CameraView
        init(_ parent: CameraView) { self.parent = parent }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                // 1. Convert image to JPEG data
                guard let imageData = uiImage.jpegData(compressionQuality: 0.8) else {
                    picker.dismiss(animated: true)
                    return
                }

                // 2. Prepare upload request
                let url = URL(string: "https://your-backend/photos/upload")! // <-- Replace with your backend
                var request = URLRequest(url: url)
                request.httpMethod = "POST"

                let boundary = UUID().uuidString
                request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

                // 3. Build multipart body
                var body = Data()
                body.append("--\(boundary)\r\n".data(using: .utf8)!)
                body.append("Content-Disposition: form-data; name=\"photo\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
                body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
                body.append(imageData)
                body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
                request.httpBody = body

                // 4. Upload asynchronously
                let task = URLSession.shared.dataTask(with: request) { data, response, error in
                    DispatchQueue.main.async {
                        if let data = data,
                        let resp = try? JSONDecoder().decode(Photo.self, from: data) {
                            // Success: Backend returns new photo (with URL + label)
                            self.parent.onCapture(resp)
                        } else {
                            // Fallback: handle error or use placeholder
                            let tempUrl = "https://via.placeholder.com/600x800.png"
                            let photo = Photo(id: UUID().uuidString, url: tempUrl, label: "Just now")
                            self.parent.onCapture(photo)
                        }
                    }
                }
                task.resume()
            }
            picker.dismiss(animated: true)
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
