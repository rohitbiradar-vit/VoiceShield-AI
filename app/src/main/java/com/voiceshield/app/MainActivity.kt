package com.voiceshield.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "VoiceShield"
        private const val SIGNALING_URL = "ws://192.168.0.106:8000/ws/room1"
        private const val AUDIO_INGESTION_URL = "ws://192.168.0.106:8000/ws/audio/room1"
    }

    private lateinit var webSocketManager: WebSocketManager
    private lateinit var webRtcManager: WebRtcManager

    // Dedicated WebSocket for streaming tapped PCM audio directly to the backend
    private var backendAudioSocket: WebSocket? = null
    private val httpClient = OkHttpClient()

    private var initialized = false

    // Screen states: "home", "incoming", "calling", "connected"
    private var currentScreen by mutableStateOf("home")
    private var callStatus by mutableStateOf("Ready")
    private var isCallConnected by mutableStateOf(false)

    // Future AI warning model placeholders (dormant until real AI models are active)
    private var showAiWarning by mutableStateOf(false)
    private var aiRiskScore by mutableStateOf("---")
    private var aiConfidence by mutableStateOf("---")

    private val microphonePermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                initializeVoiceShield()
            } else {
                callStatus = "Microphone permission required"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    VoiceShieldApp()
                }
            }
        }

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            microphonePermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        } else {
            initializeVoiceShield()
        }
    }

    private fun initializeVoiceShield() {
        if (initialized) {
            return
        }
        initialized = true

        Log.d(TAG, "Initializing VoiceShield application...")

        webRtcManager = WebRtcManager(
            context = this,
            onStateChanged = { state ->
                runOnUiThread {
                    handleRtcStateChange(state)
                }
            }
        )

        // Wire WebRTC audio tap to backend audio ingestion
        webRtcManager.setAudioSampleListener { pcmData, _, _ ->
            // Stream tapped PCM audio frames directly to backend without separate AudioRecord
            if (isCallConnected && backendAudioSocket != null) {
                try {
                    backendAudioSocket?.send(pcmData.toByteString())
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to stream audio frame to backend: ${e.message}")
                }
            }
        }

        webRtcManager.initialize()

        webSocketManager = WebSocketManager(
            serverUrl = SIGNALING_URL,
            onMessage = { message ->
                runOnUiThread {
                    handleServerMessage(message)
                }
            },
            onConnectionState = { state ->
                runOnUiThread {
                    when (state) {
                        "CONNECTED" -> {
                            if (currentScreen == "home") {
                                callStatus = "Ready"
                            }
                        }
                        "FAILED" -> {
                            if (currentScreen == "home") {
                                callStatus = "Signaling Disconnected"
                            }
                        }
                        "DISCONNECTED" -> {
                            if (currentScreen == "home") {
                                callStatus = "Disconnected"
                            }
                        }
                    }
                }
            }
        )

        webRtcManager.setSignalingManager(webSocketManager)
        webSocketManager.connect()
    }

    private fun handleRtcStateChange(state: String) {
        Log.d(TAG, "RTC State Changed: $state")

        when (state) {
            "CALLING" -> {
                currentScreen = "calling"
                callStatus = "Calling..."
            }
            "CONNECTED" -> {
                isCallConnected = true
                currentScreen = "connected"
                callStatus = "Live Audio"
                startBackendAudioIngestion()
            }
            "ENDED" -> {
                isCallConnected = false
                currentScreen = "home"
                callStatus = "Ready"
                stopBackendAudioIngestion()
            }
            "FAILED" -> {
                isCallConnected = false
                currentScreen = "home"
                callStatus = "Connection Failed"
                stopBackendAudioIngestion()
            }
        }
    }

    private fun handleServerMessage(message: String) {
        try {
            Log.d(TAG, "Signaling message received: $message")
            val json = JSONObject(message)

            when (json.optString("type")) {
                "room_ready" -> {
                    val role = json.optString("role")
                    Log.d(TAG, "Connected to room. Assigned role: $role")
                }

                "call" -> {
                    // Phone B receives incoming call notification
                    currentScreen = "incoming"
                    callStatus = "Incoming call"
                }

                "call_accepted" -> {
                    // Phone A receives acceptance from Phone B -> creates WebRTC Offer
                    Log.d(TAG, "Phone B accepted call. Initiating WebRTC offer...")
                    callStatus = "Connecting..."
                    webRtcManager.createOffer()
                }

                "call_rejected" -> {
                    callStatus = "Call Rejected"
                    currentScreen = "home"
                    isCallConnected = false
                    webRtcManager.closePeerConnection()
                    stopBackendAudioIngestion()
                }

                "offer",
                "answer",
                "candidate" -> {
                    webRtcManager.handleSignalingMessage(message)
                }

                "hangup" -> {
                    Log.d(TAG, "Peer terminated the call")
                    isCallConnected = false
                    currentScreen = "home"
                    callStatus = "Ready"
                    webRtcManager.closePeerConnection()
                    stopBackendAudioIngestion()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse incoming server message", e)
        }
    }

    // Call Actions
    private fun callFriend() {
        if (!initialized) return
        isCallConnected = false
        currentScreen = "calling"
        callStatus = "Calling..."
        webRtcManager.startOutgoingCall()
    }

    private fun acceptCall() {
        currentScreen = "calling"
        callStatus = "Connecting..."
        webRtcManager.acceptIncomingCall()
    }

    private fun rejectCall() {
        currentScreen = "home"
        callStatus = "Ready"
        webRtcManager.rejectIncomingCall()
    }

    private fun endCall() {
        isCallConnected = false
        currentScreen = "home"
        callStatus = "Ready"
        stopBackendAudioIngestion()
        webRtcManager.endCall()
    }

    // Backend Audio Stream Ingestion Connection
    private fun startBackendAudioIngestion() {
        if (backendAudioSocket != null) return

        try {
            val request = Request.Builder().url(AUDIO_INGESTION_URL).build()
            backendAudioSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                    Log.d(TAG, "Backend Audio Ingestion WebSocket Connected")
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                    Log.w(TAG, "Backend Audio Ingestion error: ${t.message}")
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Could not connect to backend audio ingestion endpoint", e)
        }
    }

    private fun stopBackendAudioIngestion() {
        try {
            backendAudioSocket?.close(1000, "Call ended")
        } catch (e: Exception) {
            Log.w(TAG, "Error closing audio websocket: ${e.message}")
        }
        backendAudioSocket = null
    }

    override fun onDestroy() {
        stopBackendAudioIngestion()

        try {
            if (::webRtcManager.isInitialized) {
                webRtcManager.close()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during WebRTC cleanup", e)
        }

        try {
            if (::webSocketManager.isInitialized) {
                webSocketManager.disconnect()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during WebSocket disconnect", e)
        }

        super.onDestroy()
    }

    // Compose User Interface
    @Composable
    private fun VoiceShieldApp() {
        when (currentScreen) {
            "home" -> HomeScreen(
                status = callStatus,
                onCall = { callFriend() }
            )
            "incoming" -> IncomingCallScreen(
                onAccept = { acceptCall() },
                onReject = { rejectCall() }
            )
            "calling" -> CallingScreen(
                status = callStatus,
                onEndCall = { endCall() }
            )
            "connected" -> ConnectedScreen(
                status = callStatus,
                isWarningActive = showAiWarning,
                riskScore = aiRiskScore,
                confidence = aiConfidence,
                onEndCall = { endCall() }
            )
        }
    }

    // 1. HOME SCREEN
    @Composable
    private fun HomeScreen(
        status: String,
        onCall: () -> Unit
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "VoiceShield",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "AI-Powered Real-Time Voice Clone Protection",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(40.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Trusted Contact",
                        fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.outline
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Friend",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.SemiBold
                    )

                    Spacer(modifier = Modifier.height(20.dp))

                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .background(
                                    color = if (status == "Ready") Color(0xFF4CAF50) else Color(0xFFFF9800),
                                    shape = RoundedCornerShape(50)
                                )
                                .padding(horizontal = 12.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = "Status: $status",
                                fontSize = 13.sp,
                                color = Color.White,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(28.dp))

                    Button(
                        onClick = onCall,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = "CALL",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }

    // 2. INCOMING CALL SCREEN
    @Composable
    private fun IncomingCallScreen(
        onAccept: () -> Unit,
        onReject: () -> Unit
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Incoming VoiceShield Call",
                fontSize = 26.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Friend",
                fontSize = 22.sp,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Medium
            )

            Spacer(modifier = Modifier.height(48.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Button(
                    onClick = onAccept,
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Text("ACCEPT", fontWeight = FontWeight.Bold)
                }

                Button(
                    onClick = onReject,
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828))
                ) {
                    Text("REJECT", fontWeight = FontWeight.Bold)
                }
            }
        }
    }

    // 3. CALLING SCREEN
    @Composable
    private fun CallingScreen(
        status: String,
        onEndCall: () -> Unit
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Calling...",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = "Friend",
                fontSize = 20.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(36.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = status,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Medium
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "Establishing secure peer-to-peer WebRTC connection...",
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.outline,
                        textAlign = TextAlign.Center
                    )
                }
            }

            Spacer(modifier = Modifier.height(48.dp))

            Button(
                onClick = onEndCall,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828))
            ) {
                Text("END CALL", fontWeight = FontWeight.Bold)
            }
        }
    }

    // 4. CONNECTED SCREEN
    @Composable
    private fun ConnectedScreen(
        status: String,
        isWarningActive: Boolean,
        riskScore: String,
        confidence: String,
        onEndCall: () -> Unit
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Call Connected",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF2E7D32)
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Status: $status",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )

            Spacer(modifier = Modifier.height(28.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "AI Protection:",
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "ACTIVE",
                            color = Color(0xFF2E7D32),
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Risk Score:",
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = riskScore,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // Architecture Warning UI Placeholder (visible only when AI triggers genuine risk)
            if (isWarningActive) {
                Spacer(modifier = Modifier.height(20.dp))
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFEBEE))
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "⚠ Potential AI-Generated Voice Detected",
                            color = Color(0xFFC62828),
                            fontWeight = FontWeight.Bold,
                            fontSize = 15.sp,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Risk Score: $riskScore | Confidence: $confidence",
                            color = Color(0xFFB71C1C),
                            fontSize = 13.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(40.dp))

            Button(
                onClick = onEndCall,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828))
            ) {
                Text("END CALL", fontWeight = FontWeight.Bold)
            }
        }
    }
}