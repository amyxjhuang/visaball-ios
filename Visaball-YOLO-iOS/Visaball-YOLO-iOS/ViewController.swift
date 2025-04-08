//
//  ViewController.swift
//  Visaball-YOLO-iOS
//
//  Created by Amy Huang on 4/7/25.
//
import UIKit
import AVFoundation
import Vision

import Cocoa

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var bufferSize: CGSize = .zero
    var inferenceTime: CFTimeInterval  = 0;
    
    private let session = AVCaptureSession()
    
    @IBOutlet weak var previewView: UIView!
    var rootLayer: CALayer! = nil
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private var detectionLayer: CALayer! = nil
    private var inferenceTimeLayer: CALayer! = nil
    private var inferenceTimeBounds: CGRect! = nil
    
    private var requests = [VNRequest]()

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }


}

