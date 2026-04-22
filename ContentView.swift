import SwiftUI

struct ContentView: View {

    @StateObject private var fmcw = FMCWEngine()
    @State private var showShare = false
    @State private var shareURLs: [URL] = []

    var body: some View {
        NavigationStack {
            VStack(spacing: 28) {

                paramsCard

                Spacer()

                // Duration counter
                Text(formatted(fmcw.duration))
                    .font(.system(size: 56, weight: .thin, design: .monospaced))
                    .foregroundStyle(fmcw.isRecording ? .primary : .tertiary)

                // Record / Stop button
                recordButton

                // Share after recording
                if fmcw.lastRecordingURL != nil {
                    shareButton
                }

                // Error display
                if let msg = fmcw.errorMessage {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("AcousticProbe")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 1) {
                        Text("AcousticProbe")
                            .font(.headline)
                        Text("FMCW 18–22 kHz")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .sheet(isPresented: $showShare) {
            ShareSheet(items: shareURLs)
        }
    }

    // MARK: - Subviews

    var paramsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Signal Parameters", systemImage: "waveform.path")
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
            Divider()
            row("Start freq",    "18 000 Hz")
            row("End freq",      "22 000 Hz")
            row("Bandwidth",     "4 kHz")
            row("Sweep time",    "20 ms / chirp")
            row("Sample rate",   "48 kHz")
            row("Amplitude",     "0.5 (−6 dBFS)")
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).fontDesign(.monospaced).bold()
        }
        .font(.subheadline)
    }

    var recordButton: some View {
        Button {
            if fmcw.isRecording {
                fmcw.stopRecording()
            } else {
                fmcw.startRecording()
            }
        } label: {
            ZStack {
                Circle()
                    .fill(fmcw.isRecording ? Color.red : Color.red.opacity(0.12))
                    .frame(width: 96, height: 96)
                if fmcw.isRecording {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(.white)
                        .frame(width: 30, height: 30)
                } else {
                    Circle()
                        .fill(.red)
                        .frame(width: 42, height: 42)
                }
            }
        }
        .symbolEffect(.pulse, isActive: fmcw.isRecording)
        .accessibilityLabel(fmcw.isRecording ? "Stop recording" : "Start recording")
    }

    var shareButton: some View {
        Button {
            if let wav = fmcw.lastRecordingURL {
                let json = wav.deletingPathExtension().appendingPathExtension("json")
                shareURLs = [wav, json].filter { FileManager.default.fileExists(atPath: $0.path) }
                showShare = true
            }
        } label: {
            Label("Export WAV + JSON", systemImage: "square.and.arrow.up")
                .font(.subheadline.bold())
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
        }
        .buttonStyle(.borderedProminent)
        .tint(.blue)
    }

    // MARK: - Helpers

    func formatted(_ t: TimeInterval) -> String {
        let m  = Int(t) / 60
        let s  = Int(t) % 60
        let ms = Int((t - Double(Int(t))) * 10)
        return String(format: "%02d:%02d.%d", m, s, ms)
    }
}

// MARK: - Share sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

#Preview {
    ContentView()
}
