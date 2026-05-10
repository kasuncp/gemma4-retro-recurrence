import Foundation

// Loads the prompts.jsonl emitted by experiments/path1_phone_mac.py
// --emit-prompts. The plan (section 5) requires byte-equivalence between
// the on-device prompt and the Mac prompt; that is achieved by bundling
// the same file the Mac script wrote.
final class PromptStore {
    enum LoadError: Error { case bundleMissing, decodeFailed(line: Int, message: String) }

    let rows: [PromptRow]

    init(rows: [PromptRow]) { self.rows = rows }

    static func loadFromBundle() throws -> PromptStore {
        guard let url = Bundle.main.url(forResource: "prompts", withExtension: "jsonl") else {
            throw LoadError.bundleMissing
        }
        return try load(from: url)
    }

    static func load(from url: URL) throws -> PromptStore {
        let data = try Data(contentsOf: url)
        guard let text = String(data: data, encoding: .utf8) else {
            throw LoadError.decodeFailed(line: 0, message: "not utf-8")
        }
        var rows: [PromptRow] = []
        let decoder = JSONDecoder()
        for (i, raw) in text.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
            let lineData = Data(raw.utf8)
            do {
                let row = try decoder.decode(PromptRow.self, from: lineData)
                rows.append(row)
            } catch {
                throw LoadError.decodeFailed(line: i + 1, message: "\(error)")
            }
        }
        return PromptStore(rows: rows)
    }

    // Returns the n distinct GSM8K problems for a cell (e.g., "C2"), in
    // index order. Mirrors the deterministic head used by the Mac runner.
    func rowsForCell(_ cell: String, n: Int) -> [PromptRow] {
        rows.filter { $0.cell == cell }
            .sorted { $0.idx < $1.idx }
            .prefix(n).map { $0 }
    }
}
