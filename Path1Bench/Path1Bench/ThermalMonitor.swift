import Foundation
import Combine

// Plan section "Per-problem measurement" step 5 + sanity check 4: track
// ProcessInfo.thermalState transitions during a run. The probe records
// every transition with its wall-clock timestamp so per-problem peak state
// is queryable by [t_start, t_end] window.
@MainActor
final class ThermalMonitor: ObservableObject {
    struct Transition: Codable {
        let t_epoch: Double
        let state: String
    }

    @Published private(set) var current: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState
    @Published private(set) var transitions: [Transition] = []

    private var observer: NSObjectProtocol?

    init() {
        appendCurrent()
        observer = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.appendCurrent() }
        }
    }

    deinit {
        if let observer = observer {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    private func appendCurrent() {
        let state = ProcessInfo.processInfo.thermalState
        current = state
        transitions.append(.init(t_epoch: Date().timeIntervalSince1970, state: Self.name(of: state)))
    }

    func reset() {
        transitions.removeAll()
        appendCurrent()
    }

    func peakInWindow(start: Double, end: Double) -> String {
        // Plan: any transition .nominal -> .fair is a soft warning; .serious
        // or .critical means real throttling. Compute the worst state
        // observed at any point in [start, end], inclusive of the state
        // active at `start`.
        let priority: [String: Int] = [
            "nominal": 0, "fair": 1, "serious": 2, "critical": 3, "unknown": 0,
        ]
        var stateAtStart = "nominal"
        for t in transitions where t.t_epoch <= start {
            stateAtStart = t.state
        }
        var worst = stateAtStart
        for t in transitions where t.t_epoch > start && t.t_epoch <= end {
            if (priority[t.state] ?? 0) > (priority[worst] ?? 0) {
                worst = t.state
            }
        }
        return worst
    }

    static func name(of state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
