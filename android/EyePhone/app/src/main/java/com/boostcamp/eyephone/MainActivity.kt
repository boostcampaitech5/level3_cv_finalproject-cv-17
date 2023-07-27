package com.boostcamp.eyephone

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.AudioManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.isInvisible
import com.boostcamp.eyephone.databinding.ActivityMainBinding
import java.io.*
import java.net.Socket
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var imgUri : Uri
    private val basePath = Environment.getExternalStorageDirectory()
    private val folderPath = "Pictures/EyePhone"
    private var imageFileName = "a.jpg"

    private var inputUrl: String = "https://m.youtube.com/"
    private lateinit var webpage: WebView
    private lateinit var audioManager : AudioManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS)
        }
        audioManager=getSystemService(AUDIO_SERVICE) as AudioManager

        // Set up the listeners for take photo and video capture buttons

        viewBinding.button.setOnClickListener {
            viewBinding.button.isInvisible = true // start 버튼 안 보이게 변경
            // 입력한 값들을 받아옴
            ip = viewBinding.inputIp.text.toString()
            port = viewBinding.inputPort.text.toString().toInt()
            inputUrl = viewBinding.inputUrl.text.toString()
            delay = viewBinding.inputDelay.text.toString().toLong()
            period = viewBinding.inputPeriod.text.toString().toLong()
            savePref() // 입력한 값 저장

            //입력한 값을 이용하여 백그라운드 작업 시작
            backGroundTask()

            // webview를 등록하고, 보이게 하고 페이지를 띄움
            webpage=findViewById(R.id.webpage)
            webpage.visibility= View.VISIBLE
            webpage.settings.apply {
                javaScriptEnabled=true
                domStorageEnabled=true
                setSupportMultipleWindows(true)
            }
            Log.d("url", inputUrl)
            webpage.apply {
                webpage.webViewClient= WebViewClient()
                webpage.webChromeClient= WebChromeClient()
                Log.d("ip", ip)
                Log.d("inputUrl", inputUrl)
                webpage.loadUrl(inputUrl)
            }
        }

        // 카메라 시작 용도
        cameraExecutor = Executors.newSingleThreadExecutor()


        // 이전에 입력해뒀던 IP, Port, 기능 시작 전 delay(웹 로딩), 사진 촬영 주기가 있으면 받아옴
        val pref = getSharedPreferences("pref", MODE_PRIVATE)
        if (pref != null){
            if(pref.contains("inputUrl")){ viewBinding.inputUrl.setText(pref.getString("inputUrl", "https://m.youtube.com/"))}
            if(pref.contains("ip")){ viewBinding.inputIp.setText(pref.getString("ip", "0.0.0.0"))}
            if(pref.contains("port")){ viewBinding.inputPort.setText(pref.getString("port", "30008"))}
            if(pref.contains("delay")){ viewBinding.inputDelay.setText(pref.getString("delay", "3000"))}
            if(pref.contains("period")){ viewBinding.inputPeriod.setText(pref.getString("period", "1000"))}
        }
    }

    private fun savePref() {
        // 입력해둔 값을 저장하기 위한 함수
        val pref = getSharedPreferences("pref", MODE_PRIVATE)
        val editor = pref.edit()
        editor.putString("ip", ip)
        editor.putString("port", port.toString())
        editor.putString("inputUrl", inputUrl)
        editor.putString("delay", delay.toString())
        editor.putString("period", period.toString())
        editor.apply()
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)  // 파일이름
            .format(System.currentTimeMillis())
        imageFileName = name
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, folderPath)
            }
        }
        // storage/emulated/0/Pictures/EyePhone 폴더에 이미지에 촬영한 날짜.jpg로 저장됨

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()
        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e("takePhoto()", "Photo capture failed: ${exc.message}", exc)
                }

                override fun
                        onImageSaved(output: ImageCapture.OutputFileResults){
                    imgUri = output.savedUri!!
                    Log.d("takePhoto()", "사진 촬영 성공")
                    photoFlag = true
                }
            }
        )
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setTargetResolution(Size(720, 1080))
                .build()

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        checkThreadEnd?.interrupt()
        mainThread?.interrupt()
        photoThread?.interrupt()
        sendThread?.interrupt()
        receiveThread?.interrupt()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "com.boostcamp.eyephone"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    add(Manifest.permission.READ_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }


    // 소켓통신에 필요한것
    private var socket: Socket? = null
    private var ip = "125.135.3.93" // 125.135.3.93 local, 118.67.129.236 대훈
    private var port = 30008

    private var checkThreadEnd:Thread? = null
    private var mainThread:Thread? = null
    private var photoThread:Thread? = null
    private var sendThread:Thread? = null
    private var receiveThread:Thread? = null
    private var scrollThread:Thread? = null

    private var photoFlag = false
    private var sendFlag = true
    private var receiveFlag = true
    private var endFlag = false
    private var autoScrollFlag = 0

    private fun backGroundTask() {
        autoTakePhotoAndConnnect()
    }

    private var scrollDirection = false
    private var scrollSpeed = 0

    private fun startScroll() {
        scrollSpeed = if(scrollDirection){ 1000 } else{ -1000 }
        webpage.scrollBy(0, scrollSpeed)
    }

    private fun autoScroll(){
        val timer = Timer()

        val timerTask: TimerTask = object : TimerTask() {
            override fun run() {
                if(autoScrollFlag == 0 || endFlag){
                    timer.cancel()
                }
                scrollSpeed = if(scrollDirection){ 1 } else{ -1 }
                webpage.scrollBy(0, scrollSpeed)
            }
        }
        timer.schedule(timerTask, 100, 1)
    }

    private var delay:Long = 3000
    private var period:Long = 5000

    //period마다 자동 촬영하는 용도
    private fun autoTakePhotoAndConnnect() {
        val timer = Timer()

        val timerTask: TimerTask = object : TimerTask() {
            override fun run() {
                if(endFlag){
                    timer.cancel()
                }
                sendFlag = true
                receiveFlag = true
                connect()
            }
        }
        timer.schedule(timerTask, delay, period)
    }

    private fun connect() {
        mainThread = object : Thread() {
            override fun run() {
                try {
                    Log.d("socket", "새 통신 시작")
                    socket = Socket(ip, port)
                    socket?.tcpNoDelay = true

                    photoThread = Thread{ takePhoto() }
                    photoThread?.start()

                    sendThread = Thread{ dataSend() }
                    sendThread?.start()

                    receiveThread = Thread{ dataReceive() }
                    receiveThread?.start()

                    Log.d("connect", "connect - thread 3개 생성")

                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }
        mainThread?.start()
    }

    private fun dataSend(){
        try {

            while(!photoFlag){}

            val filePath = "$basePath/$folderPath/$imageFileName.jpg"

            while(!File(filePath).exists()){}

            var bitmap: Bitmap? = null
            while(bitmap == null){
                try{
                    bitmap = BitmapFactory.decodeFile(filePath)
                } catch (e: Exception){ Log.e("bitmap", "비트맵 변환 시도 실패/재시도") }

            }

            val byteArray = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArray)
            val bytes = byteArray.toByteArray()

            val dos = DataOutputStream(socket?.getOutputStream())

            dos.write(bytes)
            dos.flush()
            socket?.shutdownOutput()
            Log.d("send", "이미지 송신 완료")
            File(filePath).delete()
            } catch (e: Exception){
                e.printStackTrace()
            }
        sendFlag = false
    }

    private fun dataReceive() {
        try {
            var line: String? = null

            Log.w("dataReceive", "서버 수신 시작")

            while (line == null) {
                val s = socket?.getInputStream()
                val dis = DataInputStream(s) // input에 받을꺼 넣어짐
                line = dis.read().toString()

                Log.w("dataReceive", "서버에서 제스처 $line 수신")

                when(line.toInt()){
                    1 -> {  if(autoScrollFlag == 0) { //auto 상태가 아닐때만 작동
                                scrollDirection = true // scroll down
                                startScroll() } }
                    2 -> {  if(autoScrollFlag == 0) { //auto 상태가 아닐때만 작동
                                scrollDirection=false // scroll up
                                startScroll() } }

                    3 -> {  audioManager.adjustStreamVolume( // volume down
                                AudioManager.STREAM_MUSIC,
                                AudioManager.ADJUST_LOWER,
                                AudioManager.FLAG_SHOW_UI)}
                    4 -> {  audioManager.adjustStreamVolume( // volume up
                                AudioManager.STREAM_MUSIC,
                                AudioManager.ADJUST_RAISE,
                                AudioManager.FLAG_SHOW_UI)}

                    5 -> { if(autoScrollFlag == 0) { //auto 상태가 아닐때만 작동
                            webpage.post { if (webpage.canGoBack()) { webpage.goBack() } } } }// 페이지 뒤로가기
                    6 -> { if(autoScrollFlag == 0) {
                            webpage.post { if (webpage.canGoForward()) { webpage.goForward() } } } } //페이지 앞으로 가기

                    7 -> {  if(autoScrollFlag != -1) { //이미 auto scroll down 상태가 아닐때 작동
                                scrollDirection = true
                                autoScrollFlag = -1
                                scrollThread = Thread{ autoScroll() }  // auto scroll down
                                scrollThread!!.start() } }

                    8 -> {  if(autoScrollFlag != 1) { //이미 auto scroll up 상태가 아닐때 작동
                                scrollDirection = false
                                autoScrollFlag = 1
                                scrollThread = Thread{ autoScroll() }  // auto scroll down
                                scrollThread!!.start() } }
                    9 -> {  if(autoScrollFlag != 0) { // auto scroll 상태가 아니라면
                                autoScrollFlag = 0
                                scrollThread!!.interrupt() }} //thread interrupt

                    10 -> { scrollThread!!.interrupt()
                            photoThread!!.interrupt()
                            sendThread!!.interrupt()
                            receiveThread!!.interrupt()
                            mainThread!!.interrupt()
                            endFlag = true}
                    else -> {break}
                    }

                Log.w("서버에서 받아온 값 ", "" + line)
            }
        } catch (e: Exception) { e.printStackTrace() }
        receiveFlag = false
    }
}
