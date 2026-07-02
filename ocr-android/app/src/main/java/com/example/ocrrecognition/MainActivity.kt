package com.example.ocrrecognition

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
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
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.*
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.ocrrecognition.ui.theme.OCRRecognitionTheme
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException

// Update this to the desktop's LAN IP where server.py is running.
private const val SERVER_URL = "http://<YOUR_DESKTOP_IP>:5000/predict"

private val ocrClient = OkHttpClient()

data class NormalizedRoi(val x: Float, val y: Float, val w: Float, val h: Float)

data class PredictionResponse(
    val mode: String, val prediction: String, val confidence: Float,
    val label: String, val roi: NormalizedRoi, val thresholdImage: Bitmap? = null
)

fun parsePredictionResponse(json: String): PredictionResponse? = try {
    val j = JSONObject(json)
    val r = j.getJSONObject("roi")
    PredictionResponse(
        j.getString("mode"), j.getString("prediction"),
        j.getDouble("confidence").toFloat(), j.getString("label"),
        NormalizedRoi(r.getDouble("x").toFloat(), r.getDouble("y").toFloat(),
            r.getDouble("w").toFloat(), r.getDouble("h").toFloat()),
        if (j.has("threshold_image")) Base64.decode(j.getString("threshold_image"), Base64.DEFAULT)
            .let { BitmapFactory.decodeByteArray(it, 0, it.size) } else null
    )
} catch (e: Exception) { Log.e("OCR", "Parse failed", e); null }

fun rotateBitmap(b: Bitmap, deg: Int): Bitmap =
    if (deg == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, true)

fun sendFrameToServer(jpeg: ByteArray, mode: String): PredictionResponse? {
    val body = MultipartBody.Builder().setType(MultipartBody.FORM)
        .addFormDataPart("image", "frame.jpg", jpeg.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, jpeg.size))
        .addFormDataPart("mode", mode).build()
    return try { ocrClient.newCall(Request.Builder().url(SERVER_URL).post(body).build()).execute().use { res ->
        if (!res.isSuccessful) { Log.e("OCR", "Server ${res.code}"); null } else res.body?.string()?.let { parsePredictionResponse(it) }
    }} catch (e: IOException) { Log.e("OCR", "Request failed", e); null }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OCRRecognitionTheme {
                var mode by remember { mutableStateOf("digit") }
                Scaffold(Modifier.fillMaxSize(), topBar = {
                    Surface(tonalElevation = 3.dp, modifier = Modifier.fillMaxWidth()) {
                        Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically) {
                            Text("OCR Recognition", style = MaterialTheme.typography.titleLarge)
                            Button({ mode = if (mode == "digit") "char" else "digit" }) {
                                Text(if (mode == "digit") "Switch to CHAR" else "Switch to DIGIT")
                            }
                        }
                    }
                }) { CameraScreen(Modifier.padding(it), mode) }
            }
        }
    }
}

@Composable
fun CameraScreen(modifier: Modifier = Modifier, mode: String) {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { g ->
        hasPermission = g
        if (!g) Toast.makeText(context, "Camera permission denied", Toast.LENGTH_SHORT).show()
    }
    LaunchedEffect(Unit) { if (!hasPermission) permLauncher.launch(Manifest.permission.CAMERA) }

    if (hasPermission) {
        Box(modifier.fillMaxSize()) {
            val prediction = remember { mutableStateOf<PredictionResponse?>(null) }
            val onPrediction = remember { { r: PredictionResponse -> prediction.value = r } }
            var frameSize by remember { mutableStateOf(android.util.Size(640, 480)) }
            val onFrameSize = remember { { s: android.util.Size -> frameSize = s } }
            CameraPreview(Modifier.fillMaxSize(), mode, onPrediction, onFrameSize)
            PredictionOverlay(Modifier.fillMaxSize(), prediction.value, frameSize)
        }
    } else {
        Column(modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center) {
            Text("Camera permission is required to use this app. Please grant it to continue.", textAlign = TextAlign.Center)
            Button({ permLauncher.launch(Manifest.permission.CAMERA) }, Modifier.padding(top = 16.dp)) {
                Text("Grant Camera Permission")
            }
        }
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier, mode: String,
    onPrediction: (PredictionResponse) -> Unit, onFrameSize: (android.util.Size) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var isProcessing by remember { mutableStateOf(false) }
    val analyzer = remember(mode, onPrediction, onFrameSize) {
        ImageAnalysis.Analyzer { img: ImageProxy ->
            if (isProcessing) { img.close(); return@Analyzer }
            val rot = img.imageInfo.rotationDegrees
            val w = img.width
            val h = img.height
            val plane = img.planes[0]
            val buffer = plane.buffer.rewind()
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * w
            val bmp = createBitmap(w + rowPadding / pixelStride, h, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(buffer)
            img.close()
            val cropped = if (rowPadding > 0) Bitmap.createBitmap(bmp, 0, 0, w, h) else bmp
            val rotated = rotateBitmap(cropped, rot)
            onFrameSize(android.util.Size(rotated.width, rotated.height))
            val out = ByteArrayOutputStream()
            rotated.compress(Bitmap.CompressFormat.JPEG, 85, out)
            isProcessing = true
            lifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
                val res = sendFrameToServer(out.toByteArray(), mode)
                withContext(Dispatchers.Main) { res?.let(onPrediction); isProcessing = false }
            }
        }
    }
    AndroidView(
        factory = { ctx: Context -> PreviewView(ctx).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }},
        modifier = modifier,
        update = { view: PreviewView ->
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({
                val provider = future.get()
                val resSel = ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(android.util.Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                ).build()
                val preview = CameraPreview.Builder().setResolutionSelector(resSel).build()
                    .also { it.surfaceProvider = view.surfaceProvider }
                val analysis = ImageAnalysis.Builder().setResolutionSelector(resSel)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
                    .also { it.setAnalyzer(ContextCompat.getMainExecutor(context), analyzer) }
                try {
                    provider.unbindAll()
                    provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                } catch (e: Exception) { Log.e("CameraPreview", "Binding failed", e) }
            }, ContextCompat.getMainExecutor(context))
        }
    )
}

@Composable
fun PredictionOverlay(modifier: Modifier = Modifier, response: PredictionResponse?, frameSize: android.util.Size) {
    response?.let { r ->
        BoxWithConstraints(modifier) {
            val vw = maxWidth.value
            val vh = maxHeight.value
            val fw = frameSize.width.toFloat()
            val fh = frameSize.height.toFloat()
            val scale = if (vw / fw > vh / fh) vw / fw else vh / fh
            val ox = (vw - fw * scale) / 2
            val oy = (vh - fh * scale) / 2
            val roi = r.roi
            val x = (ox + roi.x * fw * scale).dp
            val y = (oy + roi.y * fh * scale).dp
            val w = (roi.w * fw * scale).dp
            val h = (roi.h * fh * scale).dp
            Box(Modifier.offset(x = x, y = y).size(w, h).border(2.dp, Color.Green))
            Text(r.label, color = Color.Green, fontSize = 18.sp,
                modifier = Modifier.offset(x = x, y = y - 32.dp).background(Color.Black.copy(alpha = 0.6f))
                    .padding(horizontal = 6.dp, vertical = 2.dp))
            r.thresholdImage?.let { bm ->
                Image(bm.asImageBitmap(), "Thresholded ROI",
                    Modifier.align(Alignment.BottomEnd).padding(16.dp).size(120.dp).border(2.dp, Color.Green))
            }
        }
    }
}