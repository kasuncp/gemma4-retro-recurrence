import Foundation

// Writes per-problem rows as JSONL to the app's Documents directory so the
// user can retrieve them via Files.app or Xcode "Download Container".
// Filename mirrors the Mac runner's:
//     {cell.lower()}__{start:04d}_{end:04d}.jsonl
//     e.g., c2_iphone__0000_0050.jsonl
final class RunResultWriter {
    static let shared = RunResultWriter()

    private let docs: URL = {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return url
    }()

    func cellsPhoneDir() -> URL {
        let dir = docs.appendingPathComponent("cells_phone", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func filename(for cell: String, n: Int) -> URL {
        let safe = cell.lowercased().replacingOccurrences(of: "-", with: "_")
        let name = String(format: "%@__0000_%04d.jsonl", safe, n)
        return cellsPhoneDir().appendingPathComponent(name)
    }

    func sustainedFilename(for cell: String) -> URL {
        let safe = cell.lowercased().replacingOccurrences(of: "-", with: "_")
        return cellsPhoneDir().appendingPathComponent("\(safe)_sustained.jsonl")
    }

    func appendJSONLine<T: Encodable>(_ row: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = []
        let data = try encoder.encode(row)
        var line = data
        line.append(0x0a)  // newline
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.close()
        } else {
            try line.write(to: url)
        }
    }

    func reset(_ url: URL) throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}
