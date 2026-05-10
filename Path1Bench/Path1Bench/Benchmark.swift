import Foundation
import UIKit
import Combine

// Per-problem benchmark loop for one cell. Mirrors the protocol in
// plans/path_1_cot_tokens/plan8.md "Per-problem measurement":
//   1. cold-start fresh inference session (each call to ModelLoader.generate
//      builds a fresh sampler/state, no KV reuse across calls)
//   2. record prompt_tokens
//   3. start wall-clock + battery + thermal sample
//   4. generate up to 512 tokens
//   5. record peak thermal state in the per-problem window
//   6. cool down 30 s
//
// Quick answer extraction is shared with the Mac runner via the GSM8K
// "#### N" convention; we re-implement it here in Swift since prompts.jsonl
// only ships the gold integer (not the legacy regex object).
@MainActor
final class Benchmark: ObservableObject {
    enum CellID: String, CaseIterable, Identifiable {
        case c2 = "C2-iPhone"
        case a3 = "A3-iPhone"
        var id: String { rawValue }
        // Maps device-suffixed cell to the device-agnostic key used inside
        // prompts.jsonl ("C2" / "A3").
        var promptKey: String { String(rawValue.split(separator: "-").first ?? "") }
    }

    @Published private(set) var status: String = "idle"
    @Published private(set) var done: Int = 0
    @Published private(set) var total: Int = 0
    @Published private(set) var lastResult: BenchmarkResult?
    @Published private(set) var isRunning: Bool = false

    static let deviceTag = "iphone_13_pm"
    static let maxNewTokens = 512
    static let cooldownSeconds: UInt64 = 30
    static let goldHashRegex = try! NSRegularExpression(pattern: #"####\s*(-?\d+)"#)
    static let fallbackNumberRegex = try! NSRegularExpression(pattern: #"(-?\d+)"#)

    let model: ModelLoader
    let thermal: ThermalMonitor
    let battery: BatteryMonitor

    init(model: ModelLoader, thermal: ThermalMonitor, battery: BatteryMonitor) {
        self.model = model
        self.thermal = thermal
        self.battery = battery
    }

    func runCell(_ cell: CellID, n: Int, store: PromptStore, resume: Bool = false) async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        let outURL = RunResultWriter.shared.filename(for: cell.rawValue, n: n)
        if !resume {
            try? RunResultWriter.shared.reset(outURL)
        }
        let alreadyDone = (try? Self.existingIdxs(at: outURL)) ?? []
        let rows = store.rowsForCell(cell.promptKey, n: n)
        total = rows.count
        done = alreadyDone.count

        thermal.reset()
        for row in rows {
            if alreadyDone.contains(row.idx) { continue }
            status = "\(cell.rawValue) idx=\(row.idx)"
            let result = await runOne(row: row, cell: cell)
            do {
                try RunResultWriter.shared.appendJSONLine(result, to: outURL)
            } catch {
                status = "write error: \(error)"
                return
            }
            lastResult = result
            done += 1

            // Per-problem 30 s cooldown (skip after last problem).
            if row.idx != rows.last?.idx {
                try? await Task.sleep(nanoseconds: Self.cooldownSeconds * 1_000_000_000)
            }
        }
        status = "done: \(cell.rawValue), wrote \(outURL.lastPathComponent)"
    }

    private func runOne(row: PromptRow, cell: CellID) async -> BenchmarkResult {
        let tEpochStart = Date().timeIntervalSince1970
        let tStart = CACurrentMediaTime()
        let battStart = battery.currentPercent()

        var text = ""
        var nGen = 0
        var nPrompt = 0
        var elapsed = 0.0
        do {
            let r = try await model.generate(prompt: row.prompt, maxTokens: Self.maxNewTokens)
            text = r.text
            nGen = r.nGen
            nPrompt = r.nPrompt
            elapsed = r.secs
        } catch {
            text = "ERROR: \(error)"
        }
        let wallMs = (CACurrentMediaTime() - tStart) * 1000
        let tEpochEnd = Date().timeIntervalSince1970
        let battEnd = battery.currentPercent()
        let battDelta = (battStart > 0 && battEnd > 0) ? (battStart - battEnd) : nil
        let peakThermal = thermal.peakInWindow(start: tEpochStart, end: tEpochEnd)

        let (pred, hashed) = Self.extractAnswer(from: text)
        let correct = (pred != nil && row.gold != nil && pred! == row.gold!) ? 1 : 0

        return BenchmarkResult(
            idx: row.idx,
            cell: cell.rawValue,
            device: Self.deviceTag,
            gold: row.gold,
            pred: pred,
            correct: correct,
            hash_hit: hashed ? 1 : 0,
            completion: text,
            gen_tokens: nGen,
            prompt_tokens: nPrompt,
            wallclock_ms: wallMs,
            gen_secs: elapsed,
            joules_mac: nil,
            peak_temp_c_mac: nil,
            battery_delta_pct_iphone: battDelta,
            peak_thermal_state_iphone: peakThermal,
            battery_pct_start: battStart > 0 ? battStart : nil,
            battery_pct_end: battEnd > 0 ? battEnd : nil,
            t_epoch_start: tEpochStart,
            t_epoch_end: tEpochEnd
        )
    }

    static func extractAnswer(from text: String) -> (Int?, Bool) {
        let range = NSRange(text.startIndex..., in: text)
        if let m = goldHashRegex.firstMatch(in: text, range: range),
           let r = Range(m.range(at: 1), in: text),
           let v = Int(text[r]) {
            return (v, true)
        }
        // Fallback: last integer in the completion.
        var last: Int?
        let matches = fallbackNumberRegex.matches(in: text, range: range)
        if let mm = matches.last, let r = Range(mm.range(at: 1), in: text), let v = Int(text[r]) {
            last = v
        }
        return (last, false)
    }

    static func existingIdxs(at url: URL) throws -> Set<Int> {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        let data = try Data(contentsOf: url)
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        var done = Set<Int>()
        let decoder = JSONDecoder()
        for raw in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if let row = try? decoder.decode(BenchmarkResult.self, from: Data(raw.utf8)) {
                done.insert(row.idx)
            }
        }
        return done
    }
}
