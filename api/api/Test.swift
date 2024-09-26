//
//  Test.swift
//  api
//
//  Created by 田沐昕（实习） on 2024/9/26.
//

import Foundation

struct NaiveBayesModel {
    private var classLogPrior: [Double] = []
    private var featureLogProb: [[Double]] = [[]]
    private var classes: [Int] = []
    private var vocabulary: [String: Int] = [:]

    init?(directoryPath: String) {
        if let paramData = try? Data(contentsOf: Bundle.main.url(forResource: "naive_bayes_params", withExtension: "json")!),
           let vocabData = try? Data(contentsOf: Bundle.main.url(forResource: "vocabulary", withExtension: "json")!) {
            let decoder = JSONDecoder()

            if let paramStr = try? String(contentsOf: Bundle.main.url(forResource: "naive_bayes_params", withExtension: "json")!, encoding: .utf8) {
                print("param json: \(paramStr)")
            }
            
            
            if let params = try? decoder.decode(ModelParams.self, from: paramData) {
                self.classLogPrior = params.classLogPrior
                self.featureLogProb = params.featureLogProb
                self.classes = params.classes
                
                self.vocabulary = try! decoder.decode([String: Int].self, from: vocabData)
            }
        }
    }
    
    func assessRisk(for sentence: String) -> Bool {
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
        let words = sentence.lowercased().split(separator: " ")

        for word in words {
            if let index = vocabulary[String(word)] {
                features[index] = 1
            }
        }
        
        return features
    }
}

struct ModelParams: Decodable {
    let classLogPrior: [Double]
    let featureLogProb: [[Double]]
    let classes: [Int]
    
    enum CodingKeys: String, CodingKey {
        case classLogPrior = "class_log_prior"
        case featureLogProb = "feature_log_prob"
        case classes = "classes"
    }
}


