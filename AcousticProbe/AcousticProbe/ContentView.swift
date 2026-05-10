import SwiftUI

struct ContentView: View {

    @StateObject private var fmcw = FMCWEngine()
    @State private var sharePayload: SharePayload?
    @State private var recordings: [Recording] = []
    @State private var isSelecting = false
    @State private var selectedIDs: Set<String> = []

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {

                    paramsCard

                    Text(formatted(fmcw.duration))
                        .font(.system(size: 56, weight: .thin, design: .monospaced))
                        .foregroundStyle(fmcw.isRecording ? .primary : .tertiary)

                    recordButton

                    if let msg = fmcw.errorMessage {
                        Text(msg)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }

                    recordingsList
                }
                .padding()
            }
            .navigationTitle("AcousticProbe")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 1) {
                        Text("AcousticProbe").font(.headline)
                        Text("FMCW 18–22 kHz")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .sheet(item: $sharePayload) { payload in
            ShareSheet(items: payload.urls)
        }
        .onAppear { reloadRecordings() }
        .onChange(of: fmcw.lastRecordingURL) { _, _ in reloadRecordings() }
    }

    // MARK: - Subviews

    var paramsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Signal Parameters", systemImage: "waveform.path")
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
            Divider()

            // ── Frequency range ──
            paramField(label: "Start Freq (Hz)", value: $fmcw.fStart)
            paramField(label: "End Freq (Hz)", value: $fmcw.fEnd)

            // Derived info
            HStack {
                Text("Bandwidth").foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f kHz", fmcw.bandwidth / 1000))
                    .fontDesign(.monospaced).bold()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            HStack {
                Text("Range resolution").foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f cm", 34300.0 / (2 * fmcw.bandwidth)))
                    .fontDesign(.monospaced).bold()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            Divider()

            // ── Sweep duration ──
            paramFieldMs(label: "Sweep Time (ms)", value: $fmcw.sweepDuration)

            HStack {
                Text("Samples/chirp").foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(fmcw.sampleRate * fmcw.sweepDuration))")
                    .fontDesign(.monospaced).bold()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            Divider()

            // ── Amplitude ──
            paramFieldFloat(label: "Amplitude (0.1–1.0)", value: $fmcw.amplitude)

            HStack {
                Text("Level").foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f dBFS", 20 * log10(Double(fmcw.amplitude))))
                    .fontDesign(.monospaced).bold()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            if fmcw.amplitude > 0.85 {
                Text("⚠️ High amplitude — check recording for clipping")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }

            Divider()

            // ── Fixed ──
            row("Sample rate", "48 kHz")

            // ── Apply + Presets ──
            Button {
                fmcw.rebuildChirp()
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                to: nil, from: nil, for: nil)
            } label: {
                Label("Apply Changes", systemImage: "checkmark.circle.fill")
                    .font(.subheadline.bold())
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            .disabled(fmcw.isRecording)

            HStack(spacing: 8) {
                presetButton("Default", fStart: 18000, fEnd: 22000, sweep: 0.020, amp: 0.9)
                presetButton("Narrow", fStart: 19000, fEnd: 21000, sweep: 0.020, amp: 0.9)
                presetButton("Wide", fStart: 17000, fEnd: 23000, sweep: 0.020, amp: 0.8)
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Input field helpers

    func paramField(label: String, value: Binding<Double>) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary).font(.subheadline)
            Spacer()
            TextField("", value: value, format: .number)
                .keyboardType(.numberPad)
                .multilineTextAlignment(.trailing)
                .fontDesign(.monospaced)
                .font(.subheadline.bold())
                .frame(width: 100)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 6))
                .disabled(fmcw.isRecording)
        }
    }

    func paramFieldMs(label: String, value: Binding<Double>) -> some View {
        // Display and edit in ms, store in seconds
        let msBinding = Binding<Double>(
            get: { value.wrappedValue * 1000 },
            set: { value.wrappedValue = $0 / 1000 }
        )
        return HStack {
            Text(label).foregroundStyle(.secondary).font(.subheadline)
            Spacer()
            TextField("", value: msBinding, format: .number)
                .keyboardType(.decimalPad)
                .multilineTextAlignment(.trailing)
                .fontDesign(.monospaced)
                .font(.subheadline.bold())
                .frame(width: 80)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 6))
                .disabled(fmcw.isRecording)
        }
    }

    func paramFieldFloat(label: String, value: Binding<Float>) -> some View {
        let doubleBinding = Binding<Double>(
            get: { Double(value.wrappedValue) },
            set: { value.wrappedValue = Float($0) }
        )
        return HStack {
            Text(label).foregroundStyle(.secondary).font(.subheadline)
            Spacer()
            TextField("", value: doubleBinding, format: .number.precision(.fractionLength(2)))
                .keyboardType(.decimalPad)
                .multilineTextAlignment(.trailing)
                .fontDesign(.monospaced)
                .font(.subheadline.bold())
                .frame(width: 80)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 6))
                .disabled(fmcw.isRecording)
        }
    }

    // MARK: - Preset buttons

    func presetButton(_ name: String, fStart: Double, fEnd: Double,
                      sweep: Double, amp: Float) -> some View {
        Button {
            fmcw.fStart = fStart
            fmcw.fEnd = fEnd
            fmcw.sweepDuration = sweep
            fmcw.amplitude = amp
            fmcw.rebuildChirp()
        } label: {
            Text(name)
                .font(.caption2.bold())
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.blue.opacity(0.1), in: Capsule())
        }
        .disabled(fmcw.isRecording)
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

    var recordingsList: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header with Select / Refresh buttons
            HStack {
                Label("Recordings (\(recordings.count))", systemImage: "tray.full")
                    .font(.subheadline.bold())
                    .foregroundStyle(.secondary)
                Spacer()

                if !recordings.isEmpty {
                    Button {
                        withAnimation {
                            isSelecting.toggle()
                            if !isSelecting { selectedIDs.removeAll() }
                        }
                    } label: {
                        Text(isSelecting ? "Done" : "Select")
                            .font(.subheadline)
                    }
                }

                Button {
                    reloadRecordings()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.subheadline)
                }
            }

            // Multi-select action bar
            if isSelecting {
                HStack(spacing: 12) {
                    Button {
                        if selectedIDs.count == recordings.count {
                            selectedIDs.removeAll()
                        } else {
                            selectedIDs = Set(recordings.map { $0.id })
                        }
                    } label: {
                        Label(selectedIDs.count == recordings.count ? "Deselect All" : "Select All",
                              systemImage: selectedIDs.count == recordings.count ? "circle" : "checkmark.circle.fill")
                            .font(.caption)
                    }

                    Spacer()

                    Text("\(selectedIDs.count) selected")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Button {
                        shareSelected()
                    } label: {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .font(.caption.bold())
                    }
                    .disabled(selectedIDs.isEmpty)
                    .tint(.blue)

                    Button(role: .destructive) {
                        deleteSelected()
                    } label: {
                        Label("Delete", systemImage: "trash")
                            .font(.caption.bold())
                    }
                    .disabled(selectedIDs.isEmpty)
                }
                .padding(.vertical, 4)
            }

            Divider()

            if recordings.isEmpty {
                Text("No recordings yet.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 20)
            } else {
                ForEach(recordings) { rec in
                    recordingRow(rec)
                    if rec.id != recordings.last?.id {
                        Divider()
                    }
                }
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    func recordingRow(_ rec: Recording) -> some View {
        HStack(spacing: 12) {
            // Checkbox in select mode
            if isSelecting {
                Button {
                    if selectedIDs.contains(rec.id) {
                        selectedIDs.remove(rec.id)
                    } else {
                        selectedIDs.insert(rec.id)
                    }
                } label: {
                    Image(systemName: selectedIDs.contains(rec.id)
                          ? "checkmark.circle.fill" : "circle")
                        .font(.title3)
                        .foregroundStyle(selectedIDs.contains(rec.id) ? .blue : .secondary)
                }
                .buttonStyle(.borderless)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(rec.displayName)
                    .font(.subheadline.bold())
                    .fontDesign(.monospaced)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("\(rec.sizeString) · \(rec.dateString)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()

            if !isSelecting {
                Button {
                    let json = rec.wavURL.deletingPathExtension().appendingPathExtension("json")
                    let urls = [rec.wavURL, json].filter { FileManager.default.fileExists(atPath: $0.path) }
                    if !urls.isEmpty {
                        sharePayload = SharePayload(urls: urls)
                    }
                } label: {
                    Image(systemName: "square.and.arrow.up")
                        .font(.title3)
                }
                .buttonStyle(.borderless)

                Button(role: .destructive) {
                    deleteRecording(rec)
                } label: {
                    Image(systemName: "trash")
                        .font(.title3)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.borderless)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            if isSelecting {
                if selectedIDs.contains(rec.id) {
                    selectedIDs.remove(rec.id)
                } else {
                    selectedIDs.insert(rec.id)
                }
            }
        }
    }

    // MARK: - Multi-select actions

    func shareSelected() {
        let selected = recordings.filter { selectedIDs.contains($0.id) }
        var urls: [URL] = []
        for rec in selected {
            urls.append(rec.wavURL)
            let json = rec.wavURL.deletingPathExtension().appendingPathExtension("json")
            if FileManager.default.fileExists(atPath: json.path) {
                urls.append(json)
            }
        }
        if !urls.isEmpty {
            sharePayload = SharePayload(urls: urls)
        }
    }

    func deleteSelected() {
        let selected = recordings.filter { selectedIDs.contains($0.id) }
        for rec in selected {
            try? FileManager.default.removeItem(at: rec.wavURL)
            let json = rec.wavURL.deletingPathExtension().appendingPathExtension("json")
            try? FileManager.default.removeItem(at: json)
        }
        selectedIDs.removeAll()
        reloadRecordings()
    }

    // MARK: - Recordings IO

    func reloadRecordings() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        recordings = urls
            .filter { $0.pathExtension.lowercased() == "wav" }
            .compactMap { Recording(wavURL: $0) }
            .sorted { $0.modified > $1.modified }
    }

    func deleteRecording(_ rec: Recording) {
        try? FileManager.default.removeItem(at: rec.wavURL)
        let json = rec.wavURL.deletingPathExtension().appendingPathExtension("json")
        try? FileManager.default.removeItem(at: json)
        reloadRecordings()
    }

    // MARK: - Helpers

    func formatted(_ t: TimeInterval) -> String {
        let m  = Int(t) / 60
        let s  = Int(t) % 60
        let ms = Int((t - Double(Int(t))) * 10)
        return String(format: "%02d:%02d.%d", m, s, ms)
    }
}

// MARK: - Models

struct Recording: Identifiable {
    let id: String
    let wavURL: URL
    let modified: Date
    let sizeBytes: Int

    init?(wavURL: URL) {
        let attrs = try? FileManager.default.attributesOfItem(atPath: wavURL.path)
        let modified = (attrs?[.modificationDate] as? Date) ?? Date.distantPast
        let size = (attrs?[.size] as? Int) ?? 0
        self.id = wavURL.path
        self.wavURL = wavURL
        self.modified = modified
        self.sizeBytes = size
    }

    var displayName: String { wavURL.lastPathComponent }

    var sizeString: String {
        ByteCountFormatter.string(fromByteCount: Int64(sizeBytes), countStyle: .file)
    }

    var dateString: String {
        let f = DateFormatter()
        f.dateFormat = "MMM d, HH:mm:ss"
        return f.string(from: modified)
    }
}

struct SharePayload: Identifiable {
    let id = UUID()
    let urls: [URL]
}

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