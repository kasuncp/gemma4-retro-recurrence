import Foundation

// Mirrors the row format emitted by experiments/path1_phone_mac.py
// --emit-prompts. Each idx appears twice: once for cell="C2", once for
// cell="A3".
struct PromptRow: Codable, Identifiable {
    let idx: Int
    let cell: String          // "C2" or "A3"
    let prompt: String
    let prompt_tokens_hf: Int
    let gold: Int?
    let question: String

    var id: String { "\(idx)-\(cell)" }
}
