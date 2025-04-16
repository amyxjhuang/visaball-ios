//
//  ViewController.swift
//  Visaball-YOLO-iOS
//
//  Created by Amy Huang on 4/8/25.

import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    struct YOLOPrediction {
        let confidence: Float
        let boundingBox: CGRect
        let label: String
    }

    // Capture
    var bufferSize: CGSize = .zero
    var inferenceTime: CFTimeInterval  = 0;
    private let session = AVCaptureSession()
    
    // UI/Layers
    @IBOutlet weak var previewView: UIView!
    var rootLayer: CALayer! = nil
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private var detectionLayer: CALayer! = nil
    private var inferenceTimeLayer: CALayer! = nil
    private var inferenceTimeBounds: CGRect! = nil
    
    // Vision
    private var requests = [VNRequest]()
    
    // Setup
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCapture()
        setupOutput()
        setupLayers()
        try? setupVision()
        session.startRunning()
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
    
    func setupLayers() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        rootLayer = previewView.layer
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(previewLayer)
        
        inferenceTimeBounds = CGRect(x: rootLayer.frame.midX-75, y: rootLayer.frame.maxY-70, width: 150, height: 17)
        
        inferenceTimeLayer = createRectLayer(inferenceTimeBounds, [1,1,1,1])
        inferenceTimeLayer.cornerRadius = 7
        rootLayer.addSublayer(inferenceTimeLayer)
        
        detectionLayer = CALayer()
        detectionLayer.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: bufferSize.width,
                                         height: bufferSize.height)
        detectionLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionLayer)
        
        let xScale: CGFloat = rootLayer.bounds.size.width / bufferSize.height
        let yScale: CGFloat = rootLayer.bounds.size.height / bufferSize.width
        
        let scale = fmax(xScale, yScale)
    
        // rotate the layer into screen orientation and scale and mirror
        detectionLayer.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale))
        // center the layer
        detectionLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
    }
    
    func setupVision() throws {
        guard let modelURL = Bundle.main.url(forResource: "YOLOv12s-best", withExtension: "mlmodelc") else {
            throw NSError(domain: "ViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
        }
        
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    if let results = request.results {
                        self.drawResults(results)
                    }
                })
            })
            self.requests = [objectRecognition]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let rgbBuffer = convertToRGB640x640PixelBuffer(pixelBuffer) else {
            return
        }
        do {
            let model = try YOLOv12s_best(configuration: MLModelConfiguration())
            let start = CACurrentMediaTime()
            let prediction = try model.prediction(image: rgbBuffer)
            inferenceTime = (CACurrentMediaTime() - start)
            let results = postprocessYOLOOutput(prediction.var_2042, imageSize: bufferSize)
            DispatchQueue.main.async {
                self.drawYOLOPredictions(results)
            }
        } catch {
            print("Error running prediction: \(error)")
        }
    }
    
    
    func postprocessYOLOOutput(_ output: MLMultiArray, imageSize: CGSize, threshold: Float = 0.001) -> [YOLOPrediction] {
        let channels = 5
        let anchors = 8400
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(output.dataPointer))
        let values = UnsafeBufferPointer(start: ptr, count: channels * anchors)
        var results: [YOLOPrediction] = []

        for i in 0..<anchors {
            let x = values[i]
            let y = values[i + anchors]
            let w = values[i + anchors * 2]
            let h = values[i + anchors * 3]
            let conf = values[i + anchors * 4]
            print("confidence: \(conf)")
            guard conf > threshold else { continue }

            let cx = CGFloat(x) * imageSize.width
            let cy = CGFloat(y) * imageSize.height
            let width = CGFloat(w) * imageSize.width
            let height = CGFloat(h) * imageSize.height

            let rect = CGRect(x: cx - width / 2, y: cy - height / 2, width: width, height: height)
            results.append(YOLOPrediction(confidence: conf, boundingBox: rect, label: "homeplate"))
        }

        return results
    }
    
    
    func drawYOLOPredictions(_ predictions: [YOLOPrediction]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionLayer.sublayers = nil
        inferenceTimeLayer.sublayers = nil
        print("Num predictions found: \(predictions.count)")
        for prediction in predictions {
            let shapeLayer = createRectLayer(prediction.boundingBox, [0, 1, 0, 1])
            let label = NSMutableAttributedString(string: "\(prediction.label)\n\(String(format: "%.1f%%", prediction.confidence * 100))")
            let textLayer = createDetectionTextLayer(prediction.boundingBox, label)
            shapeLayer.addSublayer(textLayer)
            detectionLayer.addSublayer(shapeLayer)
        }

        let inferenceString = NSMutableAttributedString(string: String(format: "Inference time: %.1f ms", inferenceTime * 1000))
        let timeLayer = createInferenceTimeTextLayer(inferenceTimeBounds, inferenceString)
        inferenceTimeLayer.addSublayer(timeLayer)

        CATransaction.commit()
    }

    
    func drawResults(_ results: [Any]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionLayer.sublayers = nil // Clear previous detections from detectionLayer
        inferenceTimeLayer.sublayers = nil
        for observation in results {//} where observation is VNRecognizedObjectObservation {
//            let objectS = observation as? VNCoreMLFeatureValueObservation
//            let value = objectS?.featureValue
//            print("\(value))")
//            objectS.
            guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            
            // Detection with highest confidence
            let topLabelObservation = objectObservation.labels[0]
            print("Found \(topLabelObservation.identifier) with \(topLabelObservation.confidence)")
            // Rotate the bounding box into screen orientation
            let boundingBox = CGRect(origin: CGPoint(x:1.0-objectObservation.boundingBox.origin.y-objectObservation.boundingBox.size.height, y:objectObservation.boundingBox.origin.x), size: CGSize(width:objectObservation.boundingBox.size.height,height:objectObservation.boundingBox.size.width))
            
            let objectBounds = VNImageRectForNormalizedRect(boundingBox, Int(bufferSize.width), Int(bufferSize.height))
            
            let shapeLayer = createRectLayer(objectBounds, colors[topLabelObservation.identifier]!)
            
            let formattedString = NSMutableAttributedString(string: String(format: "\(topLabelObservation.identifier)\n %.1f%% ", topLabelObservation.confidence*100).capitalized)
            
            let textLayer = createDetectionTextLayer(objectBounds, formattedString)
            shapeLayer.addSublayer(textLayer)
            detectionLayer.addSublayer(shapeLayer)
        }
        
        let formattedInferenceTimeString = NSMutableAttributedString(string: String(format: "Inference time: %.1f ms ", inferenceTime*1000))
        
        let inferenceTimeTextLayer = createInferenceTimeTextLayer(inferenceTimeBounds, formattedInferenceTimeString)

        inferenceTimeLayer.addSublayer(inferenceTimeTextLayer)
        
        CATransaction.commit()
    }
        
    // Clean up capture setup
    func teardownAVCapture() {
        previewLayer.removeFromSuperlayer()
        previewLayer = nil
    }
    
    
    func convertToRGBPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // not used for now.
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)

        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &outputBuffer)

        guard let output = outputBuffer else {
            return nil
        }

        context.render(ciImage, to: output)
        return output
    }
    
    func convertToRGB640x640PixelBuffer(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // we need it to be 640x640 for now
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let resizedImage = ciImage.transformed(by: CGAffineTransform(scaleX: 640.0 / ciImage.extent.width, y: 640.0 / ciImage.extent.height))
        
        let context = CIContext()
        var outputBuffer: CVPixelBuffer?

        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferWidthKey: 640,
            kCVPixelBufferHeightKey: 640,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary

        CVPixelBufferCreate(kCFAllocatorDefault, 640, 640, kCVPixelFormatType_32BGRA, attrs, &outputBuffer)

        guard let output = outputBuffer else { return nil }

        context.render(resizedImage, to: output)
        return output
    }


}

