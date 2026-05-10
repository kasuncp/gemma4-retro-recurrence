// Path 1 plan 8 v2 - iPhone benchmark app entry point.
// See plans/path_1_cot_tokens/plan8.md for protocol.

import SwiftUI
import UIKit

@main
struct Path1BenchApp: App {
    init() {
        // Plan section "Run conditions": app holds the screen on for the
        // duration of the run; the benchmark must be the foreground task.
        UIApplication.shared.isIdleTimerDisabled = true
        // Battery monitoring must be enabled before UIDevice.batteryLevel
        // returns a real value.
        UIDevice.current.isBatteryMonitoringEnabled = true
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
