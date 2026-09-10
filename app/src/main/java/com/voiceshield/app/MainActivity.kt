package com.voiceshield.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
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
import java.util.Locale

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "VoiceShield"
        private const val PREFS_NAME = "VoiceShieldPrefs"
        private const val KEY_SERVER_HOST = "server_host"
        const val DEFAULT_SERVER_HOST = "10.20.13.184:8000"
    }

    private var serverHost by mutableStateOf(DEFAULT_SERVER_HOST)

    private fun getSignalingUrl(): String = "ws://$serverHost/ws/room1"
    private fun getAudioIngestionUrl(): String = "ws://$serverHost/ws/audio/room1"

    private fun updateServerHost(newHost: String) {
        val trimmed = newHost.trim()
            .removePrefix("ws://")
            .removePrefix("wss://")
            .removePrefix("http://")
            .removePrefix("https://")
            .trimEnd('/')
        if (trimmed.isEmpty()) return
        serverHost = trimmed
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_SERVER_HOST, trimmed).apply()
        Log.d(TAG, "Updated server host to $trimmed. Reconnecting WebSocket...")
        callStatus = "Connecting..."
        if (::webSocketManager.isInitialized) {
            webSocketManager.updateServerUrl(getSignalingUrl())
        }
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
    private var assignedRole by mutableStateOf("caller")

    // Audio capture format from WebRTC audio tap
    private var capturedSampleRate = 16000
    private var capturedChannelCount = 1

    // Live AI detection states calculated in backend
    private var showAiWarning by mutableStateOf(false)
    private var aiRiskScore by mutableStateOf("---")
    private var aiRiskValue by mutableFloatStateOf(0.0f)
    private var aiConfidence by mutableStateOf("---")
    private var aiStatus by mutableStateOf("Listening...")
    private var aiClassification by mutableStateOf("IDLE")

    // Model breakdown states for transparent live inspection
    private var vadStatusText by mutableStateOf("Waiting for speech...")
    private var aasistRiskText by mutableStateOf("---")
    private var rawnetRiskText by mutableStateOf("---")
    private var ecapaRiskText by mutableStateOf("Unenrolled (Safe Skip)")
    private var bispectralRiskText by mutableStateOf("---")
    private var aiLatencyText by mutableStateOf("---")

    // Rolling risk history for real-time Canvas graph (last 40 points)
    private val riskHistory = mutableStateListOf<Float>()

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
        webRtcManager.setAudioSampleListener { pcmData, sampleRate, channelCount ->
            capturedSampleRate = sampleRate
            capturedChannelCount = channelCount

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

        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val savedHost = prefs.getString(KEY_SERVER_HOST, null)
        serverHost = if (savedHost.isNullOrBlank() || savedHost == "10.103.52.163:8000" || savedHost == "192.168.0.106:8000") {
            prefs.edit().putString(KEY_SERVER_HOST, DEFAULT_SERVER_HOST).apply()
            DEFAULT_SERVER_HOST
        } else {
            savedHost
        }

        webSocketManager = WebSocketManager(
            serverUrl = getSignalingUrl(),
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
                        "CONNECTING" -> {
                            if (currentScreen == "home") {
                                callStatus = "Connecting..."
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
                resetAiState()
            }
            "FAILED" -> {
                isCallConnected = false
                currentScreen = "home"
                callStatus = "Connection Failed"
                stopBackendAudioIngestion()
                resetAiState()
            }
        }
    }

    private fun handleServerMessage(message: String) {
        try {
            Log.d(TAG, "Signaling message received: $message")
            val json = JSONObject(message)

            when (json.optString("type")) {
                "room_ready" -> {
                    val role = json.optString("role", "caller")
                    assignedRole = role
                    Log.d(TAG, "Connected to room. Assigned role: $assignedRole")
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
                    resetAiState()
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
                    resetAiState()
                }

                // Live AI detection verdict computed in backend
                "ai_result" -> {
                    val targetRole = json.optString("target_role")
                    if (targetRole.isNotEmpty() && targetRole != assignedRole) {
                        Log.d(TAG, "Ignoring ai_result for other role: $targetRole (my role: $assignedRole)")
                        return
                    }

                    val status = json.optString("status")
                    val classification = json.optString("classification", "IDLE")
                    aiClassification = classification
                    val latency = json.optInt("latency_ms", 0)

                    if (status == "NO_SPEECH") {
                        aiStatus = "Listening (Silence / Ambient)"
                        aiRiskScore = "Silence"
                        aiRiskValue = 0.05f
                        aiConfidence = "N/A"
                        showAiWarning = false
                        val vadProb = json.optDouble("vad_probability", 0.0)
                        vadStatusText = "No Speech (${(vadProb * 100).toInt()}%)"
                        aasistRiskText = "Gated by VAD"
                        rawnetRiskText = "Gated by VAD"
                        bispectralRiskText = "Gated by VAD"
                        aiLatencyText = "${latency}ms"

                        if (riskHistory.size >= 40) riskHistory.removeAt(0)
                        riskHistory.add(0.05f)
                    } else if (status == "ANALYZING") {
                        val risk = if (json.has("risk_score") && !json.isNull("risk_score")) json.optDouble("risk_score") else 0.05
                        val conf = if (json.has("confidence") && !json.isNull("confidence")) json.optDouble("confidence") else 0.0
                        aiRiskValue = risk.toFloat()
                        aiRiskScore = String.format(Locale.US, "%.1f%%", risk * 100.0)
                        aiConfidence = String.format(Locale.US, "%.0f%%", conf * 100.0)
                        aiStatus = "ANALYZING SPEECH..."
                        showAiWarning = false

                        if (riskHistory.size >= 40) riskHistory.removeAt(0)
                        riskHistory.add(aiRiskValue)

                        val vadProb = json.optDouble("vad_probability", 1.0)
                        vadStatusText = String.format(Locale.US, "Active Speech (%.0f%%)", vadProb * 100.0)

                        if (json.has("aasist_risk") && !json.isNull("aasist_risk")) {
                            val aRisk = json.optDouble("aasist_risk")
                            aasistRiskText = String.format(Locale.US, "Synthetic: %.1f%%", aRisk * 100.0)
                        } else {
                            aasistRiskText = "Analyzing..."
                        }

                        if (json.has("rawnet_risk") && !json.isNull("rawnet_risk")) {
                            val rRisk = json.optDouble("rawnet_risk")
                            rawnetRiskText = String.format(Locale.US, "Synthetic: %.1f%%", rRisk * 100.0)
                        } else {
                            rawnetRiskText = "Analyzing..."
                        }

                        if (json.has("speaker_risk") && !json.isNull("speaker_risk")) {
                            val sRisk = json.optDouble("speaker_risk")
                            ecapaRiskText = String.format(Locale.US, "Mismatch: %.1f%%", sRisk * 100.0)
                        } else {
                            ecapaRiskText = "Not Enrolled (Safe Skip)"
                        }

                        if (json.has("bispectral_risk") && !json.isNull("bispectral_risk")) {
                            val bRisk = json.optDouble("bispectral_risk")
                            bispectralRiskText = String.format(Locale.US, "Vocoder: %.1f%%", bRisk * 100.0)
                        } else {
                            bispectralRiskText = "Analyzing..."
                        }

                        aiLatencyText = "${latency}ms"
                    } else if (status == "ANALYZED") {
                        val risk = if (json.has("risk_score") && !json.isNull("risk_score")) json.optDouble("risk_score") else 0.0
                        val conf = if (json.has("confidence") && !json.isNull("confidence")) json.optDouble("confidence") else 0.0

                        if (risk >= 0.0) {
                            val riskPct = risk * 100.0
                            aiRiskValue = risk.toFloat()
                            aiRiskScore = String.format(Locale.US, "%.1f%%", riskPct)
                            aiConfidence = String.format(Locale.US, "%.0f%%", conf * 100.0)

                            when (classification) {
                                "LOW_RISK" -> {
                                    aiStatus = "AUTHENTIC VOICE"
                                    showAiWarning = false
                                }
                                "MEDIUM_RISK" -> {
                                    aiStatus = "SUSPICIOUS (ANOMALIES)"
                                    showAiWarning = false
                                }
                                "HIGH_RISK" -> {
                                    aiStatus = "CLONED VOICE DETECTED!"
                                    showAiWarning = true
                                }
                                else -> {
                                    aiStatus = "Analyzing..."
                                    showAiWarning = false
                                }
                            }
                        }

                        if (riskHistory.size >= 40) riskHistory.removeAt(0)
                        riskHistory.add(aiRiskValue)

                        // Sub-model breakdowns
                        val vadProb = json.optDouble("vad_probability", 1.0)
                        vadStatusText = String.format(Locale.US, "Active Speech (%.0f%%)", vadProb * 100.0)

                        if (json.has("aasist_risk") && !json.isNull("aasist_risk")) {
                            val aRisk = json.optDouble("aasist_risk")
                            aasistRiskText = String.format(Locale.US, "Synthetic: %.1f%%", aRisk * 100.0)
                        } else {
                            aasistRiskText = "N/A"
                        }

                        if (json.has("rawnet_risk") && !json.isNull("rawnet_risk")) {
                            val rRisk = json.optDouble("rawnet_risk")
                            rawnetRiskText = String.format(Locale.US, "Synthetic: %.1f%%", rRisk * 100.0)
                        } else {
                            rawnetRiskText = "N/A"
                        }

                        if (json.has("speaker_risk") && !json.isNull("speaker_risk")) {
                            val sRisk = json.optDouble("speaker_risk")
                            ecapaRiskText = String.format(Locale.US, "Mismatch: %.1f%%", sRisk * 100.0)
                        } else {
                            ecapaRiskText = "Not Enrolled (Safe Skip)"
                        }

                        if (json.has("bispectral_risk") && !json.isNull("bispectral_risk")) {
                            val bRisk = json.optDouble("bispectral_risk")
                            bispectralRiskText = String.format(Locale.US, "Vocoder: %.1f%%", bRisk * 100.0)
                        } else {
                            bispectralRiskText = "N/A"
                        }

                        aiLatencyText = "${latency}ms"
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse incoming server message", e)
        }
    }

    private fun resetAiState() {
        showAiWarning = false
        aiRiskScore = "---"
        aiRiskValue = 0.0f
        aiConfidence = "---"
        aiStatus = "Listening..."
        aiClassification = "IDLE"
        vadStatusText = "Waiting for speech..."
        aasistRiskText = "---"
        rawnetRiskText = "---"
        ecapaRiskText = "Unenrolled (Safe Skip)"
        bispectralRiskText = "---"
        aiLatencyText = "---"
        riskHistory.clear()
    }

    // Call Actions
    private fun callFriend() {
        if (!initialized) return
        isCallConnected = false
        resetAiState()
        currentScreen = "calling"
        callStatus = "Calling..."
        webRtcManager.startOutgoingCall()
    }

    private fun acceptCall() {
        resetAiState()
        currentScreen = "calling"
        callStatus = "Connecting..."
        webRtcManager.acceptIncomingCall()
    }

    private fun rejectCall() {
        currentScreen = "home"
        callStatus = "Ready"
        resetAiState()
        webRtcManager.rejectIncomingCall()
    }

    private fun endCall() {
        isCallConnected = false
        currentScreen = "home"
        callStatus = "Ready"
        stopBackendAudioIngestion()
        resetAiState()
        webRtcManager.endCall()
    }

    // Backend Audio Stream Ingestion Connection
    private fun startBackendAudioIngestion() {
        if (backendAudioSocket != null) return

        try {
            val urlWithParams = "${getAudioIngestionUrl()}/$assignedRole?sample_rate=$capturedSampleRate&channels=$capturedChannelCount"
            Log.d(TAG, "Connecting to audio ingestion endpoint: $urlWithParams (role=$assignedRole)")
            val request = Request.Builder().url(urlWithParams).build()
            backendAudioSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                    Log.d(TAG, "Backend Audio Ingestion Connected: role=$assignedRole, sr=$capturedSampleRate, ch=$capturedChannelCount")
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
                currentServerHost = serverHost,
                onUpdateServerHost = { updateServerHost(it) },
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
                role = assignedRole,
                status = callStatus,
                isWarningActive = showAiWarning,
                riskScore = aiRiskScore,
                riskValue = aiRiskValue,
                confidence = aiConfidence,
                aiStatus = aiStatus,
                classification = aiClassification,
                vadStatus = vadStatusText,
                aasistRisk = aasistRiskText,
                rawnetRisk = rawnetRiskText,
                ecapaRisk = ecapaRiskText,
                bispectralRisk = bispectralRiskText,
                latency = aiLatencyText,
                riskHistory = riskHistory,
                onEndCall = { endCall() }
            )
        }
    }

    // 1. HOME SCREEN
    @Composable
    private fun HomeScreen(
        status: String,
        currentServerHost: String,
        onUpdateServerHost: (String) -> Unit,
        onCall: () -> Unit
    ) {
        var inputHost by remember(currentServerHost) { mutableStateOf(currentServerHost) }
        val scrollState = rememberScrollState()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "VoiceShield",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = "AI-Powered Real-Time Voice Clone Protection",
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(28.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Trusted Contact",
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.outline
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Friend",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.SemiBold
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    val (statusColor, statusBg) = when (status) {
                        "Ready" -> Pair(Color(0xFF2E7D32), Color(0xFFE8F5E9))
                        "Connecting..." -> Pair(Color(0xFFF57C00), Color(0xFFFFF3E0))
                        else -> Pair(Color(0xFFC62828), Color(0xFFFFEBEE))
                    }

                    Box(
                        modifier = Modifier
                            .background(
                                color = statusBg,
                                shape = RoundedCornerShape(50)
                            )
                            .padding(horizontal = 14.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = "Status: $status",
                            fontSize = 12.sp,
                            color = statusColor,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Spacer(modifier = Modifier.height(20.dp))

                    Button(
                        onClick = onCall,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(12.dp),
                        enabled = (status == "Ready")
                    ) {
                        Text(
                            text = "CALL",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            // BACKEND SERVER CONFIGURATION
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(18.dp)
                ) {
                    Text(
                        text = "BACKEND SERVER CONFIGURATION",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.outline
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    OutlinedTextField(
                        value = inputHost,
                        onValueChange = { inputHost = it },
                        label = { Text("Server Host (IP:Port)") },
                        placeholder = { Text(DEFAULT_SERVER_HOST) },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Button(
                            onClick = { onUpdateServerHost(inputHost) },
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text("Reconnect", fontWeight = FontWeight.Bold)
                        }

                        OutlinedButton(
                            onClick = {
                                inputHost = DEFAULT_SERVER_HOST
                                onUpdateServerHost(DEFAULT_SERVER_HOST)
                            },
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text("Reset")
                        }
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

    // 4. CONNECTED SCREEN — LIVE AI VOICE CLONING DETECTION DASHBOARD
    @Composable
    private fun ConnectedScreen(
        role: String,
        status: String,
        isWarningActive: Boolean,
        riskScore: String,
        riskValue: Float,
        confidence: String,
        aiStatus: String,
        classification: String,
        vadStatus: String,
        aasistRisk: String,
        rawnetRisk: String,
        ecapaRisk: String,
        bispectralRisk: String,
        latency: String,
        riskHistory: List<Float>,
        onEndCall: () -> Unit
    ) {
        val scrollState = rememberScrollState()

        // Dynamic risk accent color styling based on real calculated classification
        val (riskBadgeBg, riskBadgeText, riskAccentColor) = when (classification) {
            "HIGH_RISK" -> Triple(Color(0xFFFFEBEE), Color(0xFFC62828), Color(0xFFD32F2F))
            "MEDIUM_RISK" -> Triple(Color(0xFFFFF3E0), Color(0xFFE65100), Color(0xFFF57C00))
            "LOW_RISK" -> Triple(Color(0xFFE8F5E9), Color(0xFF2E7D32), Color(0xFF388E3C))
            "ANALYZING" -> Triple(Color(0xFFE0F7FA), Color(0xFF00838F), Color(0xFF00ACC1))
            "SILENCE" -> Triple(Color(0xFFF5F5F5), Color(0xFF616161), Color(0xFF9E9E9E))
            else -> Triple(Color(0xFFE3F2FD), Color(0xFF1565C0), Color(0xFF1976D2))
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // Call Header
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF2E7D32))
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "WebRTC Live Call • $status",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    color = Color(0xFF2E7D32)
                )
            }

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = if (role == "caller") "Friend (Receiver)" else "Friend (Caller)",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(2.dp))

            Text(
                text = "Analyzing Incoming Voice: ${if (role == "caller") "Receiver (Peer)" else "Caller (Peer)"}",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Medium
            )

            Spacer(modifier = Modifier.height(18.dp))

            // PRIMARY AI RISK CARD
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(18.dp)
                ) {
                    // Header row: Title + Status Badge
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "AI DEFENSE • ${role.uppercase(Locale.US)}",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.outline
                        )

                        Box(
                            modifier = Modifier
                                .background(color = riskBadgeBg, shape = RoundedCornerShape(50))
                                .padding(horizontal = 10.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = aiStatus,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                color = riskBadgeText
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    // Large Risk Score Display
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.Bottom,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column {
                            Text(
                                text = "Risk Score",
                                fontSize = 13.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = riskScore,
                                fontSize = 38.sp,
                                fontWeight = FontWeight.ExtraBold,
                                color = riskAccentColor
                            )
                        }

                        Column(horizontalAlignment = Alignment.End) {
                            Text(
                                text = "Confidence",
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.outline
                            )
                            Text(
                                text = confidence,
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    // Dynamic Linear Risk Meter
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp))
                            .background(Color(0xFFE0E0E0))
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(fraction = riskValue.coerceIn(0.02f, 1.0f))
                                .height(8.dp)
                                .clip(RoundedCornerShape(4.dp))
                                .background(riskAccentColor)
                        )
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    // Latency / Pipeline Health
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Window: 1.5s (200ms hop)",
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.outline
                        )
                        Text(
                            text = "Latency: $latency",
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // REAL-TIME ROLLING RISK SCORE GRAPH (CANVAS)
            RiskScoreGraph(
                history = riskHistory,
                currentRisk = riskValue
            )

            Spacer(modifier = Modifier.height(14.dp))

            // MULTI-BRANCH NEURAL MODEL BREAKDOWN CARD
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(18.dp)
                ) {
                    Text(
                        text = "MULTI-BRANCH NEURAL INSPECTION",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.outline
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "Analyzing incoming peer voice via AudioTrack tap",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(modifier = Modifier.height(14.dp))

                    // Model 0: Silero VAD
                    ModelRow(
                        modelName = "Silero VAD",
                        description = "Speech Activity Gate",
                        value = vadStatus,
                        valueColor = if (vadStatus.contains("Active")) Color(0xFF2E7D32) else Color(0xFF757575)
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    // Model 1: AASIST (Lightweight Graph Anti-Spoofing)
                    ModelRow(
                        modelName = "AASIST",
                        description = "Spectral Graph Anti-Spoofing",
                        value = aasistRisk,
                        valueColor = MaterialTheme.colorScheme.onSurface
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    // Model 2: RawNet3
                    ModelRow(
                        modelName = "RawNet3",
                        description = "Sinc-CNN Anti-Spoofing",
                        value = rawnetRisk,
                        valueColor = MaterialTheme.colorScheme.onSurface
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    // Model 3: ECAPA-TDNN
                    ModelRow(
                        modelName = "ECAPA-TDNN",
                        description = "Speaker ID Verification",
                        value = ecapaRisk,
                        valueColor = MaterialTheme.colorScheme.onSurface
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    // Model 4: Bispectral Analyzer
                    ModelRow(
                        modelName = "Bispectral Analyzer",
                        description = "Vocoder Phase Artifacts",
                        value = bispectralRisk,
                        valueColor = MaterialTheme.colorScheme.onSurface
                    )
                }
            }

            // URGENT HIGH-RISK WARNING BANNER
            if (isWarningActive) {
                Spacer(modifier = Modifier.height(14.dp))
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .border(1.5.dp, Color(0xFFD32F2F), RoundedCornerShape(16.dp)),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFEBEE)),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "⚠ HIGH RISK OF VOICE CLONING ATTACK",
                            color = Color(0xFFC62828),
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Synthetic vocoder artifacts detected in the voice stream. Do NOT share passwords, banking OTPs, or authorize money transfers.",
                            color = Color(0xFFB71C1C),
                            fontSize = 12.sp,
                            textAlign = TextAlign.Center,
                            lineHeight = 16.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(28.dp))

            // END CALL BUTTON
            Button(
                onClick = onEndCall,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828))
            ) {
                Text("END CALL", fontWeight = FontWeight.Bold, fontSize = 16.sp)
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }

    @Composable
    private fun RiskScoreGraph(
        history: List<Float>,
        currentRisk: Float,
        modifier: Modifier = Modifier
    ) {
        Card(
            modifier = modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "REAL-TIME RISK TRAJECTORY",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.outline
                    )
                    Text(
                        text = "200ms Hop Cadence",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.outline
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color(0xFF181824))
                ) {
                    val width = size.width
                    val height = size.height
                    val paddingLeft = 16f
                    val paddingRight = 16f
                    val paddingTop = 12f
                    val paddingBottom = 12f
                    val chartWidth = width - paddingLeft - paddingRight
                    val chartHeight = height - paddingTop - paddingBottom

                    // Threshold horizontal heights
                    val y100 = paddingTop
                    val y70 = paddingTop + chartHeight * (1.0f - 0.70f)
                    val y40 = paddingTop + chartHeight * (1.0f - 0.40f)
                    val y0 = paddingTop + chartHeight

                    // Colored Background Threshold Bands
                    // Red Band: 70% - 100% (High Risk)
                    drawRect(
                        color = Color(0x30E53935),
                        topLeft = Offset(paddingLeft, y100),
                        size = Size(chartWidth, y70 - y100)
                    )
                    // Orange Band: 40% - 70% (Medium Risk)
                    drawRect(
                        color = Color(0x28FB8C00),
                        topLeft = Offset(paddingLeft, y70),
                        size = Size(chartWidth, y40 - y70)
                    )
                    // Green Band: 0% - 40% (Low Risk)
                    drawRect(
                        color = Color(0x2043A047),
                        topLeft = Offset(paddingLeft, y40),
                        size = Size(chartWidth, y0 - y40)
                    )

                    // Threshold reference lines
                    drawLine(
                        color = Color(0x66E53935),
                        start = Offset(paddingLeft, y70),
                        end = Offset(width - paddingRight, y70),
                        strokeWidth = 1.5f
                    )
                    drawLine(
                        color = Color(0x66FB8C00),
                        start = Offset(paddingLeft, y40),
                        end = Offset(width - paddingRight, y40),
                        strokeWidth = 1.5f
                    )

                    // Draw rolling history line
                    if (history.isNotEmpty()) {
                        val points = history.takeLast(40)
                        val stepX = if (points.size > 1) chartWidth / (points.size - 1) else chartWidth
                        val path = Path()

                        points.forEachIndexed { index, value ->
                            val clamped = value.coerceIn(0.0f, 1.0f)
                            val x = paddingLeft + index * stepX
                            val y = paddingTop + chartHeight * (1.0f - clamped)

                            if (index == 0) {
                                path.moveTo(x, y)
                            } else {
                                path.lineTo(x, y)
                            }
                        }

                        val strokeColor = when {
                            currentRisk >= 0.70f -> Color(0xFFEF5350)
                            currentRisk >= 0.40f -> Color(0xFFFFA726)
                            else -> Color(0xFF66BB6A)
                        }

                        drawPath(
                            path = path,
                            color = strokeColor,
                            style = Stroke(width = 3.5f)
                        )

                        // Glowing point on latest observation
                        val lastClamped = (points.lastOrNull() ?: currentRisk).coerceIn(0.0f, 1.0f)
                        val lastX = paddingLeft + (points.size - 1) * stepX
                        val lastY = paddingTop + chartHeight * (1.0f - lastClamped)

                        drawCircle(
                            color = strokeColor.copy(alpha = 0.35f),
                            radius = 8f,
                            center = Offset(lastX, lastY)
                        )
                        drawCircle(
                            color = strokeColor,
                            radius = 4.5f,
                            center = Offset(lastX, lastY)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(6.dp))

                // Threshold legend
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("0-40% Safe", fontSize = 10.sp, color = Color(0xFF43A047), fontWeight = FontWeight.SemiBold)
                    Text("40-70% Caution", fontSize = 10.sp, color = Color(0xFFFB8C00), fontWeight = FontWeight.SemiBold)
                    Text("70-100% Danger", fontSize = 10.sp, color = Color(0xFFE53935), fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }

    @Composable
    private fun ModelRow(
        modelName: String,
        description: String,
        value: String,
        valueColor: Color
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = modelName,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = description,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.outline
                )
            }
            Text(
                text = value,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = valueColor,
                textAlign = TextAlign.End
            )
        }
    }
}