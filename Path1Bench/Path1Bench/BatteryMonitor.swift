import Foundation
import UIKit
import Combine

// Plan section "Power (best-effort)" + sanity checks 6 & 7: estimate joules
// indirectly via UIDevice.batteryLevel deltas. The runner samples level at
// the start and end of each problem and stores the delta in percentage
// points. The analyze script applies the per-cell baseline subtraction.
@MainActor
final class BatteryMonitor: ObservableObject {
    @Published private(set) var level: Double = 1.0      // 0.0...1.0 or -1.0 if unknown
    @Published private(set) var stateName: String = "unknown"
    @Published private(set) var idleBaselinePctPerSec: Double? = nil

    private var levelObserver: NSObjectProtocol?
    private var stateObserver: NSObjectProtocol?

    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        readNow()
        levelObserver = NotificationCenter.default.addObserver(
            forName: UIDevice.batteryLevelDidChangeNotification,
            object: nil, queue: .main
        ) { [weak self] _ in Task { @MainActor in self?.readNow() } }
        stateObserver = NotificationCenter.default.addObserver(
            forName: UIDevice.batteryStateDidChangeNotification,
            object: nil, queue: .main
        ) { [weak self] _ in Task { @MainActor in self?.readNow() } }
    }

    deinit {
        if let o = levelObserver { NotificationCenter.default.removeObserver(o) }
        if let o = stateObserver { NotificationCenter.default.removeObserver(o) }
    }

    func readNow() {
        let lvl = Double(UIDevice.current.batteryLevel)
        level = lvl
        stateName = Self.name(of: UIDevice.current.batteryState)
    }

    func currentPercent() -> Double {
        let l = Double(UIDevice.current.batteryLevel)
        return l < 0 ? -1.0 : l * 100.0
    }

    // Sanity check 6: 5-minute idle baseline. Subtracting this makes the
    // inference-attributable battery delta meaningful.
    func calibrateIdleBaseline(durationSeconds: Double) async throws -> Double {
        let start = currentPercent()
        let t0 = Date().timeIntervalSince1970
        try await Task.sleep(nanoseconds: UInt64(durationSeconds * 1e9))
        let end = currentPercent()
        let elapsed = Date().timeIntervalSince1970 - t0
        let rate = (start - end) / elapsed   // %/sec, positive = drain
        idleBaselinePctPerSec = rate
        return rate
    }

    static func name(of state: UIDevice.BatteryState) -> String {
        switch state {
        case .unknown:   return "unknown"
        case .unplugged: return "unplugged"
        case .charging:  return "charging"
        case .full:      return "full"
        @unknown default: return "unknown"
        }
    }
}
