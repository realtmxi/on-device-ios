import Foundation

public struct NaiveBayesModel {
    private var classLogPrior: [Double]
    private var featureLogProb: [[Double]]
    private var classes: [Int]
    private var vocabulary: [String: Int] // Maps words to feature indices

    public init?() {
        guard let url = Bundle.main.url(forResource: "naive_bayes_params", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let params = try? JSONDecoder().decode(ModelParams.self, from: data),
              let vocabUrl = Bundle.main.url(forResource: "vocabulary", withExtension: "json"),
              let vocabData = try? Data(contentsOf: vocabUrl),
              let vocab = try? JSONDecoder().decode([String: Int].self, from: vocabData) else {
            return nil
        }
        
        self.classLogPrior = params.classLogPrior
        self.featureLogProb = params.featureLogProb
        self.classes = params.classes
        self.vocabulary = vocab
    }
    
    public func assessRisk(for sentence: String) -> Bool {
        let inputFeatures = extractFeatures(from: sentence)
        
        var logProbabilities = classLogPrior
        
        for (i, feature) in inputFeatures.enumerated() {
            for (classIndex, _) in classes.enumerated() {
                logProbabilities[classIndex] += featureLogProb[classIndex][i] * Double(feature)
            }
        }
        
        let predictedClass = logProbabilities.enumerated().max(by: { $0.element < $1.element })?.offset
        return classes[predictedClass ?? 0] == 1
    }

    private func extractFeatures(from sentence: String) -> [Int] {
        var features = Array(repeating: 0, count: vocabulary.count)
        
        // Tokenize sentence into words
        let words = sentence.split(separator: " ").map { $0.lowercased() }
        
        for word in words {
            if let index = vocabulary[word] {
                features[index] = 1
            }
        }
        
        return features
    }
}

struct ModelParams: Codable {
    let classLogPrior: [Double]
    let featureLogProb: [[Double]]
    let classes: [Int]
}
