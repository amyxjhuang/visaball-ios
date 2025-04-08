//
//  ViewController.swift
//  Visaball-YOLO-iOS
//
//  Created by Amy Huang on 4/7/25.
//  Reference: https://medium.com/better-programming/how-to-build-a-yolov5-object-detection-app-on-ios-39c8c77dfe58
import UIKit
import AVFoundation
import Vision

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
        setupCapture()
        setupOutput()
        setupLayers()
        try? setupVision()
        session.startRunning()
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }

    func setupCapture() {
           var deviceInput: AVCaptureDeviceInput!
           let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
           do {
               deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
           } catch {
               print("Could not create video device input: \(error)")
               return
           }
           
           session.beginConfiguration()
           session.sessionPreset = .vga640x480
           
           guard session.canAddInput(deviceInput) else {
               print("Could not add video device input to the session")
               session.commitConfiguration()
               return
           }
           session.addInput(deviceInput)
           
           do {
               try  videoDevice!.lockForConfiguration()
               let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
               bufferSize.width = CGFloat(dimensions.width)
               bufferSize.height = CGFloat(dimensions.height)
               videoDevice!.unlockForConfiguration()
           } catch {
               print(error)
           }
           session.commitConfiguration()
       }
       
    func setupOutput() {
           let videoDataOutput = AVCaptureVideoDataOutput()
           let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
           
           if session.canAddOutput(videoDataOutput) {
               session.addOutput(videoDataOutput)
               videoDataOutput.alwaysDiscardsLateVideoFrames = true
               videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
               videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
           } else {
               print("Could not add video data output to the session")
               session.commitConfiguration()
               return
           }
       }
}

