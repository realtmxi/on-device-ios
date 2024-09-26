//
//  ViewController.swift
//  api
//
//  Created by 田沐昕（实习） on 2024/9/26.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        // Example Usage

        let directoryPath = "/Users/tianmuxin/desktop/on-device-ios"

        if let model = NaiveBayesModel(directoryPath: directoryPath) {
            let sentence = "How doing today"
            let isRisk = model.assessRisk(for: sentence)
            print("Privacy Leak Risk Detected: \(isRisk)")
        } else {
            print("Failed to initialize the model.")
        }
        
    }
    
    


}

