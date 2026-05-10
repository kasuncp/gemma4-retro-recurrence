import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// Wraps mlx-swift / mlx-swift-examples for the on-device Gemma-4 E2B 4-bit
// run. Per the user's choice, weights come from HuggingFace on first
// launch (see Path1Bench/README.md - the user must convert + upload the
// MLX 4-bit weights, then set defaultRepoID to that path).
//
// === API ADJUSTMENT POINT ===
// The exact surface of MLXLLM / MLXLMCommon shifts across mlx-swift-examples
// tags. This file is the only place that touches it; if the build fails on
// the `loadContainer` or `generate` call, fix HERE.
@MainActor
final class ModelLoader: ObservableObject {
    enum State: Equatable {
        case idle
        case downloading(Double)
        case loading
        case ready
        case failed(String)
    }

    static let defaultRepoID = "mlx-community/gemma-4-E2B-it-4bit-mlx"

    @Published private(set) var state: State = .idle
    @Published var repoID: String = UserDefaults.standard.string(forKey: "Path1Bench.repoID")
        ?? ModelLoader.defaultRepoID

    private(set) var container: ModelContainer?

    func saveRepoID(_ id: String) {
        repoID = id
        UserDefaults.standard.set(id, forKey: "Path1Bench.repoID")
    }

    func load() async {
        state = .downloading(0)
        do {
            let configuration = ModelConfiguration(id: repoID)
            let factory = LLMModelFactory.shared
            container = try await factory.loadContainer(
                configuration: configuration
            ) { progress in
                Task { @MainActor [weak self] in
                    self?.state = .downloading(progress.fractionCompleted)
                }
            }
            state = .ready
        } catch {
            state = .failed("\(error)")
        }
    }

    struct GenResult {
        let text: String
        let nGen: Int
        let nPrompt: Int
        let secs: Double
    }

    func generate(prompt: String, maxTokens: Int) async throws -> GenResult {
        guard let container = container else {
            throw NSError(
                domain: "Path1Bench", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "model not loaded"]
            )
        }
        let t0 = Date().timeIntervalSince1970

        var nGen = 0
        var nPrompt = 0
        var collected = ""

        try await container.perform { context in
            let userInput = UserInput(prompt: prompt)
            let lmInput = try await context.processor.prepare(input: userInput)
            nPrompt = lmInput.text.tokens.size
            let parameters = GenerateParameters(temperature: 0.0)
            let stream = try MLXLMCommon.generate(
                input: lmInput, parameters: parameters, context: context
            )
            for await event in stream {
                switch event {
                case .chunk(let str):
                    collected += str
                case .info(let info):
                    nGen = info.generationTokenCount
                @unknown default:
                    continue
                }
                if nGen >= maxTokens { break }
            }
        }
        let elapsed = Date().timeIntervalSince1970 - t0
        return GenResult(text: collected, nGen: nGen, nPrompt: nPrompt, secs: elapsed)
    }
}
