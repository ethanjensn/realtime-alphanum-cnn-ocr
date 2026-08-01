package com.example.ocrrecognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.createBitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class OcrInterpreter(context: Context) {

    private val digitInterpreter: Interpreter
    private val charInterpreter: Interpreter
    private val emnistMapping: Map<Int, Char>

    init {
        OpenCVLoader.initLocal()
        Log.d("OcrInterpreter", "OpenCV loaded")

        val digitModel = loadMappedAsset(context, "digits_model.tflite")
        val charModel = loadMappedAsset(context, "chars_and_digits_model.tflite")

        val options = Interpreter.Options().apply { setNumThreads(4) }
        digitInterpreter = Interpreter(digitModel, options)
        charInterpreter = Interpreter(charModel, options)

        emnistMapping = loadEmnistMapping(context)
        Log.d("OcrInterpreter", "Loaded ${emnistMapping.size} EMNIST mappings")
    }

    private fun loadMappedAsset(context: Context, assetName: String): MappedByteBuffer {
        val assetFd = context.assets.openFd(assetName)
        val fileChannel = FileInputStream(assetFd.fileDescriptor).channel
        val buffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
        fileChannel.close()
        assetFd.close()
        return buffer
    }

    private fun loadEmnistMapping(context: Context): Map<Int, Char> {
        val mapping = mutableMapOf<Int, Char>()
        val reader = BufferedReader(InputStreamReader(context.assets.open("emnist-byclass-mapping.txt")))
        reader.forEachLine { line ->
            val parts = line.trim().split(" ")
            if (parts.size == 2) {
                val idx = parts[0].toInt()
                val unicodeInt = parts[1].toInt()
                mapping[idx] = unicodeInt.toChar()
            }
        }
        reader.close()
        return mapping
    }

    private fun preprocess(roiBitmap: Bitmap): Pair<FloatArray, Bitmap> {
        val gray = Mat()
        Utils.bitmapToMat(roiBitmap, gray)
        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGBA2GRAY)

        val resizedHigh = Mat()
        Imgproc.resize(gray, resizedHigh, Size(84.0, 84.0), 0.0, 0.0, Imgproc.INTER_CUBIC)

        val smoothed = Mat()
        Imgproc.bilateralFilter(resizedHigh, smoothed, 9, 75.0, 75.0)

        val thresh = Mat()
        Imgproc.threshold(smoothed, thresh, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

        val resized28 = Mat()
        Imgproc.resize(thresh, resized28, Size(28.0, 28.0), 0.0, 0.0, Imgproc.INTER_AREA)

        val normalized = FloatArray(28 * 28)
        val pixelData = ByteArray(28 * 28)
        resized28.get(0, 0, pixelData)
        for (i in pixelData.indices) {
            normalized[i] = (pixelData[i].toInt() and 0xFF) / 255.0f
        }

        val displayBitmap = createBitmap(200, 200)
        val displayMat = Mat()
        Imgproc.resize(resized28, displayMat, Size(200.0, 200.0), 0.0, 0.0, Imgproc.INTER_NEAREST)
        Imgproc.cvtColor(displayMat, displayMat, Imgproc.COLOR_GRAY2RGBA)
        Utils.matToBitmap(displayMat, displayBitmap)

        gray.release()
        resizedHigh.release()
        smoothed.release()
        thresh.release()
        resized28.release()
        displayMat.release()

        return Pair(normalized, displayBitmap)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (l in logits) if (l > maxLogit) maxLogit = l
        var sum = 0.0
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            exps[i] = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += exps[i]
        }
        for (i in exps.indices) exps[i] = (exps[i] / sum).toFloat()
        return exps
    }

    private fun argmax(arr: FloatArray): Int {
        var maxIdx = 0
        for (i in arr.indices) if (arr[i] > arr[maxIdx]) maxIdx = i
        return maxIdx
    }

    fun predict(roiBitmap: Bitmap, mode: String, roi: NormalizedRoi): PredictionResponse {
        val (normalized, thresholdBitmap) = preprocess(roiBitmap)

        val predLabel: String
        val confidence: Float
        val predClass: Int
        val label: String

        if (mode == "char") {
            val input = Array(1) { Array(28) { Array(28) { FloatArray(1) } } }
            for (i in 0 until 28) {
                for (j in 0 until 28) {
                    input[0][i][j][0] = normalized[i * 28 + j]
                }
            }
            val output = Array(1) { FloatArray(62) }
            charInterpreter.run(input, output)

            val probabilities = softmax(output[0])
            confidence = probabilities[argmax(probabilities)]
            predClass = argmax(probabilities)
            predLabel = emnistMapping[predClass]?.toString() ?: "?"

            val aConfidence = probabilities[36]
            if (aConfidence > 2.0f) {
                val finalPredLabel = "a"
                val modeText = "CHAR"
                val finalLabel = "$modeText: $finalPredLabel (${"%.2f".format(aConfidence)})"
                return PredictionResponse(
                    mode, finalPredLabel, aConfidence, finalLabel, roi, thresholdBitmap
                )
            }

            val confidenceThreshold = 0.1f
            val modeText = "CHAR"
            label = if (confidence > confidenceThreshold) {
                "$modeText: $predLabel (${"%.2f".format(confidence)})"
            } else {
                "No ${modeText.lowercase()} detected"
            }
        } else {
            val input = Array(1) { Array(28) { Array(28) { FloatArray(1) } } }
            for (i in 0 until 28) {
                for (j in 0 until 28) {
                    input[0][i][j][0] = normalized[i * 28 + j]
                }
            }
            val output = Array(1) { FloatArray(10) }
            digitInterpreter.run(input, output)

            val probabilities = output[0]
            confidence = probabilities[argmax(probabilities)]
            predClass = argmax(probabilities)
            predLabel = predClass.toString()

            val confidenceThreshold = 0.7f
            val modeText = "DIGIT"
            label = if (confidence > confidenceThreshold) {
                "$modeText: $predLabel (${"%.2f".format(confidence)})"
            } else {
                "No ${modeText.lowercase()} detected"
            }
        }

        return PredictionResponse(
            mode, predLabel, confidence, label, roi, thresholdBitmap
        )
    }

    fun close() {
        digitInterpreter.close()
        charInterpreter.close()
    }
}
