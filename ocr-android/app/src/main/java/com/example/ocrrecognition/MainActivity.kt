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
import androidx.camera.core.Camera
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
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

private val ocrClient = OkHttpClient()

data class NormalizedRoi(val x: Float, val y: Float, val w: Float, val h: Float)

private enum class Corner { TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT }

private fun adjustRoiForCorner(
    roi: NormalizedRoi, corner: Corner, ndx: Float, ndy: Float
): NormalizedRoi = when (corner) {
    Corner.TOP_LEFT -> {
        val nx = (roi.x + ndx).coerceIn(0f, roi.x + roi.w - 0.05f)
        val ny = (roi.y + ndy).coerceIn(0f, roi.y + roi.h - 0.05f)
        roi.copy(x = nx, y = ny, w = roi.w - (nx - roi.x), h = roi.h - (ny - roi.y))
    }
    Corner.TOP_RIGHT -> {
        val ny = (roi.y + ndy).coerceIn(0f, roi.y + roi.h - 0.05f)
        val nw = (roi.w + ndx).coerceIn(0.05f, 1f - roi.x)
        roi.copy(y = ny, h = roi.h - (ny - roi.y), w = nw)
    }
    Corner.BOTTOM_LEFT -> {
        val nx = (roi.x + ndx).coerceIn(0f, roi.x + roi.w - 0.05f)
        val nh = (roi.h + ndy).coerceIn(0.05f, 1f - roi.y)
        roi.copy(x = nx, w = roi.w - (nx - roi.x), h = nh)
    }
    Corner.BOTTOM_RIGHT -> {
        val nw = (roi.w + ndx).coerceIn(0.05f, 1f - roi.x)
        val nh = (roi.h + ndy).coerceIn(0.05f, 1f - roi.y)
        roi.copy(w = nw, h = nh)
    }
}

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

fun sendFrameToServer(jpeg: ByteArray, mode: String, roi: NormalizedRoi): PredictionResponse? {
    val body = MultipartBody.Builder().setType(MultipartBody.FORM)
        .addFormDataPart("image", "frame.jpg", jpeg.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, jpeg.size))
        .addFormDataPart("mode", mode)
        .addFormDataPart("roi_x", roi.x.toString())
        .addFormDataPart("roi_y", roi.y.toString())
        .addFormDataPart("roi_w", roi.w.toString())
        .addFormDataPart("roi_h", roi.h.toString())
        .build()
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
            var userRoi by remember { mutableStateOf(NormalizedRoi(0.42f, 0.36f, 0.16f, 0.28f)) }
            var zoomRatio by remember { mutableFloatStateOf(1f) }
            var camera by remember { mutableStateOf<Camera?>(null) }
            CameraPreview(
                Modifier.fillMaxSize(), mode, onPrediction, onFrameSize,
                userRoi, { camera = it }
            )
            PredictionOverlay(
                Modifier.fillMaxSize(), prediction.value, frameSize,
                userRoi, { userRoi = it }, zoomRatio, { zoomRatio = it }, camera
            )
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
    onPrediction: (PredictionResponse) -> Unit, onFrameSize: (android.util.Size) -> Unit,
    userRoi: NormalizedRoi, onCameraReady: (Camera) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var isProcessing by remember { mutableStateOf(false) }
    val currentRoi = rememberUpdatedState(userRoi)
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
                val res = sendFrameToServer(out.toByteArray(), mode, currentRoi.value)
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
                    val cam = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                    onCameraReady(cam)
                } catch (e: Exception) { Log.e("CameraPreview", "Binding failed", e) }
            }, ContextCompat.getMainExecutor(context))
        }
    )
}

@Composable
private fun CornerHandle(
    cx: Float, cy: Float,
    onDrag: (Float, Float) -> Unit
) {
    val currentOnDrag = rememberUpdatedState(onDrag)
    Box(
        Modifier
            .offset(x = (cx - 14).dp, y = (cy - 14).dp)
            .size(28.dp)
            .background(Color.White)
            .border(2.dp, Color.Green)
            .pointerInput(Unit) {
                detectDragGestures { _, dragAmount ->
                    currentOnDrag.value(dragAmount.x, dragAmount.y)
                }
            }
    )
}

@Composable
fun PredictionOverlay(
    modifier: Modifier = Modifier, response: PredictionResponse?, frameSize: android.util.Size,
    userRoi: NormalizedRoi, onRoiChange: (NormalizedRoi) -> Unit,
    zoomRatio: Float, onZoomChange: (Float) -> Unit, camera: Camera?
) {
    val maxZoom = camera?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1f
    val currentRoi = rememberUpdatedState(userRoi)
    val currentZoom = rememberUpdatedState(zoomRatio)

    BoxWithConstraints(
        modifier.pointerInput(maxZoom) {
            detectTransformGestures { _, _, scale, _ ->
                val newZoom = (currentZoom.value * scale).coerceIn(1f, maxZoom)
                onZoomChange(newZoom)
                camera?.cameraControl?.setZoomRatio(newZoom)
            }
        }
    ) {
        val vw = maxWidth.value
        val vh = maxHeight.value
        val fw = frameSize.width.toFloat()
        val fh = frameSize.height.toFloat()
        val scale = if (vw / fw > vh / fh) vw / fw else vh / fh
        val ox = (vw - fw * scale) / 2
        val oy = (vh - fh * scale) / 2

        val boxX = ox + currentRoi.value.x * fw * scale
        val boxY = oy + currentRoi.value.y * fh * scale
        val boxW = currentRoi.value.w * fw * scale
        val boxH = currentRoi.value.h * fh * scale

        Box(
            Modifier
                .offset(x = boxX.dp, y = boxY.dp)
                .size(boxW.dp, boxH.dp)
                .border(2.dp, Color.Green)
                .pointerInput(Unit) {
                    detectDragGestures { _, dragAmount ->
                        val ndx = dragAmount.x / (fw * scale)
                        val ndy = dragAmount.y / (fh * scale)
                        val r = currentRoi.value
                        onRoiChange(r.copy(
                            x = (r.x + ndx).coerceIn(0f, 1f - r.w),
                            y = (r.y + ndy).coerceIn(0f, 1f - r.h)
                        ))
                    }
                }
        )

        CornerHandle(boxX, boxY) { dx, dy ->
            onRoiChange(adjustRoiForCorner(currentRoi.value, Corner.TOP_LEFT, dx / (fw * scale), dy / (fh * scale)))
        }
        CornerHandle(boxX + boxW, boxY) { dx, dy ->
            onRoiChange(adjustRoiForCorner(currentRoi.value, Corner.TOP_RIGHT, dx / (fw * scale), dy / (fh * scale)))
        }
        CornerHandle(boxX, boxY + boxH) { dx, dy ->
            onRoiChange(adjustRoiForCorner(currentRoi.value, Corner.BOTTOM_LEFT, dx / (fw * scale), dy / (fh * scale)))
        }
        CornerHandle(boxX + boxW, boxY + boxH) { dx, dy ->
            onRoiChange(adjustRoiForCorner(currentRoi.value, Corner.BOTTOM_RIGHT, dx / (fw * scale), dy / (fh * scale)))
        }

        response?.let { r ->
            Text(r.label, color = Color.Green, fontSize = 18.sp,
                modifier = Modifier.offset(x = boxX.dp, y = (boxY - 32).dp)
                    .background(Color.Black.copy(alpha = 0.6f))
                    .padding(horizontal = 6.dp, vertical = 2.dp))
            r.thresholdImage?.let { bm ->
                Image(
                    bitmap = bm.asImageBitmap(),
                    contentDescription = "Thresholded ROI",
                    modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp).size(120.dp).border(2.dp, Color.Green)
                )
            }
        }
    }
}