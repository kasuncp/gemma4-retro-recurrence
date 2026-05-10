import Foundation
import UIKit

// Plan section "Sustained-thermal probe": run continuous generation for
// 5 minutes (no cooldown). Sample tokens/sec at 1 Hz buckets, record
// every thermalState transition, and capture battery delta over the run.
@MainActor
final class SustainedProbe: ObservableObject {
    @Published private(set) var status: String = "idle"
    @Published private(set) var elapsed: Double = 0
    @Published private(set) var tokensPerSec: Double = 0
    @Published private(set) var totalTokens: Int = 0
    @Published private(set) var thermalState: String = "nominal"
    @Published private(set) var isRunning: Bool = false

    let model: ModelLoader
    let thermal: ThermalMonitor
    let battery: BatteryMonitor

    init(model: ModelLoader, thermal: ThermalMonitor, battery: BatteryMonitor) {
        self.model = model
        self.thermal = thermal
        self.battery = battery
    }

    func run(cell: Benchmark.CellID, durationSeconds: Double, store: PromptStore) async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        let outURL = RunResultWriter.shared.sustainedFilename(for: cell.rawValue)
        try? RunResultWriter.shared.reset(outURL)

        let prompts = store.rowsForCell(cell.promptKey, n: 50)
        guard !prompts.isEmpty else { status = "no prompts"; return }

        thermal.reset()
        let battStart = battery.currentPercent()
        let t0 = CACurrentMediaTime()
        var bucketStart = t0
        var bucketIdx = 0
        var bucketTokens = 0
        var total = 0
        var pi = 0

        while CACurrentMediaTime() - t0 < durationSeconds {
            let prompt = prompts[pi % prompts.count]
            pi += 1
            do {
                let r = try await model.generate(prompt: prompt.prompt, maxTokens: Benchmark.maxNewTokens)
                bucketTokens += r.nGen
                total += r.nGen
            } catch {
                status = "gen error: \(error)"
                break
            }
            let now = CACurrentMediaTime()
            while now - bucketStart >= 1.0 {
                let bucket = SustainedBucket(
                    cell: cell.rawValue,
                    device: Benchmark.deviceTag,
                    bucket_idx: bucketIdx,
                    elapsed_s: Double(bucketIdx),
                    tokens_in_bucket: bucketTokens,
                    total_tokens: total,
                    tokens_per_sec: Double(bucketTokens),
                    thermal_state_at_bucket: ThermalMonitor.name(of: ProcessInfo.processInfo.thermalState),
                    battery_pct_at_bucket: battery.currentPercent() > 0 ? battery.currentPercent() : nil
                )
                try? RunResultWriter.shared.appendJSONLine(bucket, to: outURL)
                bucketIdx += 1
                bucketStart += 1.0
                bucketTokens = 0
                tokensPerSec = bucket.tokens_per_sec
                totalTokens = total
                elapsed = Double(bucketIdx)
                thermalState = bucket.thermal_state_at_bucket
            }
        }

        let battEnd = battery.currentPercent()
        let summary: [String: Any] = [
            "cell": cell.rawValue,
            "device": Benchmark.deviceTag,
            "summary": true,
            "duration_s": durationSeconds,
            "total_tokens": total,
            "battery_pct_start": battStart,
            "battery_pct_end": battEnd,
            "battery_delta_pct": (battStart > 0 && battEnd > 0) ? (battStart - battEnd) : NSNull(),
            "thermal_transitions": thermal.transitions.map {
                ["t_epoch": $0.t_epoch, "state": $0.state]
            } as [Any],
        ]
        if let json = try? JSONSerialization.data(withJSONObject: summary, options: []) {
            var line = json
            line.append(0x0a)
            if FileManager.default.fileExists(atPath: outURL.path),
               let h = try? FileHandle(forWritingTo: outURL) {
                try? h.seekToEnd()
                try? h.write(contentsOf: line)
                try? h.close()
            } else {
                try? line.write(to: outURL)
            }
        }

        status = "done: \(total) tokens / \(Int(durationSeconds))s"
    }
}
