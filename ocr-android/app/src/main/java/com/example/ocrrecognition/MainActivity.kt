package com.example.ocrrecognition

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview as CameraPreview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.ocrrecognition.ui.theme.OCRRecognitionTheme
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

// Update this to the desktop's LAN IP where server.py is running.
private const val SERVER_URL = "http://<YOUR_DESKTOP_IP>:5000/predict"

private val ocrClient = OkHttpClient()

data class NormalizedRoi(
    val x: Float,
    val y: Float,
    val w: Float,
    val h: Float
)

data class PredictionResponse(
    val mode: String,
    val prediction: String,
    val confidence: Float,
    val label: String,
    val roi: NormalizedRoi,
    val thresholdImage: Bitmap? = null
)

fun parsePredictionResponse(jsonString: String): PredictionResponse? {
    return try {
        val json = JSONObject(jsonString)
        val roiJson = json.getJSONObject("roi")
        val thresholdImage = if (json.has("threshold_image")) {
            val bytes = Base64.decode(json.getString("threshold_image"), Base64.DEFAULT)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } else null
        PredictionResponse(
            mode = json.getString("mode"),
            prediction = json.getString("prediction"),
            confidence = json.getDouble("confidence").toFloat(),
            label = json.getString("label"),
            roi = NormalizedRoi(
                x = roiJson.getDouble("x").toFloat(),
                y = roiJson.getDouble("y").toFloat(),
                w = roiJson.getDouble("w").toFloat(),
                h = roiJson.getDouble("h").toFloat()
            ),
            thresholdImage = thresholdImage
        )
    } catch (e: Exception) {
        Log.e("OCR", "Failed to parse prediction response", e)
        null
    }
}

fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return bitmap
    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

fun sendFrameToServer(jpegBytes: ByteArray, mode: String): PredictionResponse? {
    val requestBody = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart(
            "image",
            "frame.jpg",
            jpegBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, jpegBytes.size)
        )
        .addFormDataPart("mode", mode)
        .build()

    val request = Request.Builder()
        .url(SERVER_URL)
        .post(requestBody)
        .build()

    return try {
        ocrClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                Log.e("OCR", "Server returned ${response.code}")
                return null
            }
            val body = response.body?.string()
            if (body != null) parsePredictionResponse(body) else null
        }
    } catch (e: IOException) {
        Log.e("OCR", "Request to server failed", e)
        null
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OCRRecognitionTheme {
                var mode by remember { mutableStateOf("digit") }
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        Surface(
                            tonalElevation = 3.dp,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 8.dp, vertical = 12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "OCR Recognition",
                                    style = MaterialTheme.typography.titleLarge
                                )
                                Button(
                                    onClick = {
                                        mode = if (mode == "digit") "char" else "digit"
                                    }
                                ) {
                                    Text(
                                        text = if (mode == "digit") "Switch to CHAR" else "Switch to DIGIT"
                                    )
                                }
                            }
                        }
                    }
                ) { innerPadding ->
                    CameraScreen(
                        modifier = Modifier.padding(innerPadding),
                        mode = mode
                    )
                }
            }
        }
    }
}

@Composable
fun CameraScreen(modifier: Modifier = Modifier, mode: String) {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        hasPermission = isGranted
        if (!isGranted) {
            Toast.makeText(context, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    LaunchedEffect(Unit) {
        if (!hasPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (hasPermission) {
        Box(modifier = modifier.fillMaxSize()) {
            val predictionResponse = remember { mutableStateOf<PredictionResponse?>(null) }
            val onPrediction = remember {
                { response: PredictionResponse ->
                    predictionResponse.value = response
                }
            }

            var frameSize by remember { mutableStateOf(android.util.Size(640, 480)) }
            val onFrameSize = remember {
                { size: android.util.Size ->
                    frameSize = size
                }
            }

            CameraPreview(
                modifier = Modifier.fillMaxSize(),
                mode = mode,
                onPrediction = onPrediction,
                onFrameSize = onFrameSize
            )

            PredictionOverlay(
                modifier = Modifier.fillMaxSize(),
                response = predictionResponse.value,
                frameSize = frameSize
            )
        }
    } else {
        PermissionDeniedScreen(
            modifier = modifier.fillMaxSize(),
            onRequestPermission = {
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        )
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    mode: String,
    onPrediction: (PredictionResponse) -> Unit,
    onFrameSize: (android.util.Size) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var isProcessing by remember { mutableStateOf(false) }

    val analyzer = remember(mode, onPrediction, onFrameSize) {
        ImageAnalysis.Analyzer { imageProxy: ImageProxy ->
            if (isProcessing) {
                imageProxy.close()
                return@Analyzer
            }

            val rotation = imageProxy.imageInfo.rotationDegrees
            val sensorBitmap = createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
            imageProxy.planes[0].buffer.rewind()
            sensorBitmap.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            imageProxy.close()

            val bitmap = rotateBitmap(sensorBitmap, rotation)
            onFrameSize(android.util.Size(bitmap.width, bitmap.height))

            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream)
            val jpegBytes = stream.toByteArray()

            isProcessing = true
            lifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
                val response = sendFrameToServer(jpegBytes, mode)
                withContext(Dispatchers.Main) {
                    response?.let(onPrediction)
                    isProcessing = false
                }
            }
        }
    }

    AndroidView(
        factory = { ctx: Context ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            }
        },
        modifier = modifier,
        update = { view: PreviewView ->
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val resolutionSelector = ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            android.util.Size(640, 480),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()

                val preview = CameraPreview.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .build()
                    .also { it.surfaceProvider = view.surfaceProvider }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { it.setAnalyzer(ContextCompat.getMainExecutor(context), analyzer) }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    Log.e("CameraPreview", "Camera binding failed", e)
                }
            }, ContextCompat.getMainExecutor(context))
        }
    )
}

@Composable
fun PredictionOverlay(
    modifier: Modifier = Modifier,
    response: PredictionResponse?,
    frameSize: android.util.Size
) {
    response?.let { r ->
        BoxWithConstraints(modifier = modifier) {
            val viewWidth = maxWidth.value
            val viewHeight = maxHeight.value
            val frameWidth = frameSize.width.toFloat()
            val frameHeight = frameSize.height.toFloat()
            val scale = if (viewWidth / frameWidth > viewHeight / frameHeight) {
                viewWidth / frameWidth
            } else {
                viewHeight / frameHeight
            }
            val visibleWidth = frameWidth * scale
            val visibleHeight = frameHeight * scale
            val offsetX = (viewWidth - visibleWidth) / 2
            val offsetY = (viewHeight - visibleHeight) / 2

            val roi = r.roi
            val x = (offsetX + roi.x * frameWidth * scale).dp
            val y = (offsetY + roi.y * frameHeight * scale).dp
            val w = (roi.w * frameWidth * scale).dp
            val h = (roi.h * frameHeight * scale).dp

            Box(
                modifier = Modifier
                    .offset(x = x, y = y)
                    .size(width = w, height = h)
                    .border(2.dp, Color.Green)
            )

            Text(
                text = r.label,
                color = Color.Green,
                fontSize = 18.sp,
                modifier = Modifier
                    .offset(x = x, y = y - 32.dp)
                    .background(Color.Black.copy(alpha = 0.6f))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            )

            r.thresholdImage?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Thresholded ROI",
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(16.dp)
                        .size(120.dp)
                        .border(2.dp, Color.Green)
                )
            }
        }
    }
}

@Composable
fun PermissionDeniedScreen(
    modifier: Modifier = Modifier,
    onRequestPermission: () -> Unit
) {
    Column(
        modifier = modifier.padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Camera permission is required to use this app. Please grant it to continue.",
            textAlign = TextAlign.Center
        )
        Button(
            onClick = onRequestPermission,
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text("Grant Camera Permission")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun PermissionDeniedScreenPreview() {
    OCRRecognitionTheme {
        PermissionDeniedScreen(onRequestPermission = {})
    }
}