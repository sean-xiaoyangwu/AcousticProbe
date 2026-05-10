import AVFoundation
import Combine

class FMCWEngine: ObservableObject {

    // FMCW parameters — all configurable from UI
    let sampleRate: Double = 48000  // fixed by hardware
    @Published var fStart: Double = 18000       // Hz
    @Published var fEnd: Double = 22000         // Hz
    @Published var sweepDuration: Double = 0.020 // seconds per chirp
    @Published var amplitude: Float = 0.9       // 0.0–1.0

    // Derived (read-only)
    var bandwidth: Double { fEnd - fStart }
    var sweepDurationMs: Double { sweepDuration * 1000 }

    @Published var isRecording = false
    @Published var duration: TimeInterval = 0
    @Published var lastRecordingURL: URL?
    @Published var errorMessage: String?

    private var engine: AVAudioEngine!
    private var player: AVAudioPlayerNode!
    private var recordFile: AVAudioFile?
    private var chirpBuffer: AVAudioPCMBuffer!
    private var durationTimer: Timer?
    private var startDate: Date?

    init() {
        configureSession()
        chirpBuffer = buildChirp()
    }

    /// Rebuild chirp when any parameter changes
    func rebuildChirp() {
        // Validate ranges
        fStart = max(1000, min(fStart, 23000))
        fEnd = max(fStart + 500, min(fEnd, 24000))
        sweepDuration = max(0.005, min(sweepDuration, 0.100))
        amplitude = max(0.1, min(amplitude, 1.0))
        chirpBuffer = buildChirp()
    }

    // MARK: - Audio session

    private func configureSession() {
        let s = AVAudioSession.sharedInstance()
        try? s.setCategory(.playAndRecord,
                           mode: .measurement,
                           options: .defaultToSpeaker)
        try? s.setPreferredSampleRate(sampleRate)
        try? s.setPreferredIOBufferDuration(0.005)
        try? s.setActive(true)
    }

    // MARK: - Chirp generation (linear FM)

    private func buildChirp() -> AVAudioPCMBuffer {
        let N = Int(sampleRate * sweepDuration)
        let B = fEnd - fStart
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(N))!
        buf.frameLength = AVAudioFrameCount(N)
        let ptr = buf.floatChannelData![0]
        for i in 0 ..< N {
            let t = Double(i) / sampleRate
            let phi = 2 * Double.pi * (fStart * t + (B / (2 * sweepDuration)) * t * t)
            ptr[i] = Float(sin(phi)) * amplitude
        }
        return buf
    }

    // MARK: - Recording control

    func startRecording() {
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                if granted {
                    self?.doStart()
                } else {
                    self?.errorMessage = "Microphone access denied — enable in Settings"
                }
            }
        }
    }

    private func doStart() {
        guard !isRecording else { return }
        errorMessage = nil

        // Fresh engine each session avoids state issues
        engine = AVAudioEngine()
        player = AVAudioPlayerNode()
        engine.attach(player)

        let playFmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.connect(player, to: engine.mainMixerNode, format: playFmt)

        let inputFmt = engine.inputNode.outputFormat(forBus: 0)

        // Create WAV output file
        let ts = Int(Date().timeIntervalSince1970)
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let wavURL = dir.appendingPathComponent("fmcw_\(ts).wav")

        do {
            recordFile = try AVAudioFile(forWriting: wavURL, settings: inputFmt.settings)
        } catch {
            errorMessage = "File error: \(error.localizedDescription)"
            return
        }

        // Tap microphone
        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFmt) { [weak self] buf, _ in
            try? self?.recordFile?.write(from: buf)
        }

        do {
            try engine.start()
        } catch {
            errorMessage = "Engine error: \(error.localizedDescription)"
            return
        }

        // Loop chirp indefinitely
        player.scheduleBuffer(chirpBuffer, at: nil, options: .loops)
        player.play()

        isRecording = true
        startDate = Date()
        durationTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self, let s = self.startDate else { return }
            DispatchQueue.main.async { self.duration = Date().timeIntervalSince(s) }
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        player?.stop()
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        durationTimer?.invalidate()
        durationTimer = nil

        let wavURL = recordFile?.url
        recordFile = nil
        isRecording = false

        if let url = wavURL {
            lastRecordingURL = url
            writeParamsJSON(near: url)
        }
    }

    // MARK: - Save parameters alongside WAV

    private func writeParamsJSON(near wavURL: URL) {
        let params: [String: Any] = [
            "sample_rate_hz": sampleRate,
            "f_start_hz": fStart,
            "f_end_hz": fEnd,
            "sweep_duration_s": sweepDuration,
            "sweep_duration_ms": sweepDuration * 1000,
            "bandwidth_hz": bandwidth,
            "amplitude": amplitude,
            "amplitude_dBFS": 20 * log10(Double(amplitude)),
            "range_resolution_cm": round(34300.0 / (2 * bandwidth) * 100) / 100
        ]
        let jsonURL = wavURL.deletingPathExtension().appendingPathExtension("json")
        if let data = try? JSONSerialization.data(withJSONObject: params, options: .prettyPrinted) {
            try? data.write(to: jsonURL)
        }
    }
}