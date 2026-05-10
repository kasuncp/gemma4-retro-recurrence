import Foundation

// One row per (idx, cell) emitted by the iPhone runner. Schema mirrors the
// Mac runner's JSONL row (plan section "Deliverables") so the analyze step
// can merge across devices.
struct BenchmarkResult: Codable {
    let idx: Int
    let cell: String          // "C2-iPhone" or "A3-iPhone"
    let device: String        // "iphone_13_pm"
    let gold: Int?
    let pred: Int?
    let correct: Int
    let hash_hit: Int
    let completion: String
    let gen_tokens: Int
    let prompt_tokens: Int
    let wallclock_ms: Double
    let gen_secs: Double
    let joules_mac: Double?            // always nil on iPhone
    let peak_temp_c_mac: Double?       // always nil on iPhone
    let battery_delta_pct_iphone: Double?
    let peak_thermal_state_iphone: String?  // "nominal"|"fair"|"serious"|"critical"
    let battery_pct_start: Double?
    let battery_pct_end: Double?
    let t_epoch_start: Double
    let t_epoch_end: Double
}

struct SustainedBucket: Codable {
    let cell: String
    let device: String
    let bucket_idx: Int
    let elapsed_s: Double
    let tokens_in_bucket: Int
    let total_tokens: Int
    let tokens_per_sec: Double
    let thermal_state_at_bucket: String
    let battery_pct_at_bucket: Double?
}
