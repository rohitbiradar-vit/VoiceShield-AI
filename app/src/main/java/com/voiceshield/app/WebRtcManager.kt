package com.voiceshield.app

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.webrtc.AudioSource
import org.webrtc.AudioTrack
import org.webrtc.IceCandidate
import org.webrtc.MediaConstraints
import org.webrtc.PeerConnection
import org.webrtc.PeerConnectionFactory
import org.webrtc.RtpReceiver
import org.webrtc.SessionDescription
import org.webrtc.audio.JavaAudioDeviceModule

class WebRtcManager(
    private val context: Context,
    private val onStateChanged: (String) -> Unit
) {

    companion object {
        private const val TAG = "VoiceShieldRTC"
        private const val AUDIO_TRACK_ID = "VoiceShieldAudioTrack"
        private const val STREAM_ID = "VoiceShieldStream"

        @Volatile
        private var isFactoryInitialized = false
    }

    private var factory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var audioDeviceModule: JavaAudioDeviceModule? = null
    private var audioSource: AudioSource? = null
    private var audioTrack: AudioTrack? = null
    private var signaling: WebSocketManager? = null
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? android.media.AudioManager

    private val pendingCandidates = mutableListOf<IceCandidate>()

    @Volatile
    private var remoteDescriptionSet = false

    @Volatile
    private var initialized = false

    @Volatile
    private var isCaller = false

    @Volatile
    private var closed = false

    // Tap callback for audio ingestion without duplicate AudioRecord
    private var onAudioSampleCaptured: ((ByteArray, Int, Int) -> Unit)? = null

    fun setSignalingManager(manager: WebSocketManager) {
        signaling = manager
    }

    fun setAudioSampleListener(listener: (ByteArray, Int, Int) -> Unit) {
        onAudioSampleCaptured = listener
    }

    fun initialize() {
        if (initialized || closed) {
            return
        }

        try {
            Log.d(TAG, "Initializing WebRTC PeerConnectionFactory...")

            synchronized(Companion) {
                if (!isFactoryInitialized) {
                    val initOptions = PeerConnectionFactory.InitializationOptions
                        .builder(context.applicationContext)
                        .setEnableInternalTracer(false)
                        .createInitializationOptions()
                    PeerConnectionFactory.initialize(initOptions)
                    isFactoryInitialized = true
                }
            }

            // Conservative audio settings: hardware AEC and NS disabled to prevent native driver crashes
            val admBuilder = JavaAudioDeviceModule.builder(context.applicationContext)
                .setUseHardwareAcousticEchoCanceler(false)
                .setUseHardwareNoiseSuppressor(false)
                .setAudioRecordErrorCallback(object : JavaAudioDeviceModule.AudioRecordErrorCallback {
                    override fun onWebRtcAudioRecordInitError(errorMessage: String) {
                        Log.e(TAG, "WebRTC AudioRecord init error: $errorMessage")
                    }

                    override fun onWebRtcAudioRecordStartError(
                        errorCode: JavaAudioDeviceModule.AudioRecordStartErrorCode,
                        errorMessage: String
                    ) {
                        Log.e(TAG, "WebRTC AudioRecord start error: $errorCode ($errorMessage)")
                    }

                    override fun onWebRtcAudioRecordError(errorMessage: String) {
                        Log.e(TAG, "WebRTC AudioRecord error: $errorMessage")
                    }
                })
                .setAudioTrackErrorCallback(object : JavaAudioDeviceModule.AudioTrackErrorCallback {
                    override fun onWebRtcAudioTrackInitError(errorMessage: String) {
                        Log.e(TAG, "WebRTC AudioTrack init error: $errorMessage")
                    }

                    override fun onWebRtcAudioTrackStartError(
                        errorCode: JavaAudioDeviceModule.AudioTrackStartErrorCode,
                        errorMessage: String
                    ) {
                        Log.e(TAG, "WebRTC AudioTrack start error: $errorCode ($errorMessage)")
                    }

                    override fun onWebRtcAudioTrackError(errorMessage: String) {
                        Log.e(TAG, "WebRTC AudioTrack error: $errorMessage")
                    }
                })
                .setSamplesReadyCallback { audioSamples ->
                    // Official WebRTC audio tap callback. Direct raw PCM bytes from the call capture.
                    onAudioSampleCaptured?.invoke(
                        audioSamples.data,
                        audioSamples.sampleRate,
                        audioSamples.channelCount
                    )
                }

            audioDeviceModule = admBuilder.createAudioDeviceModule()

            factory = PeerConnectionFactory.builder()
                .setAudioDeviceModule(audioDeviceModule)
                .createPeerConnectionFactory()

            initialized = true
            Log.d(TAG, "WebRTC successfully initialized")

        } catch (e: Exception) {
            initialized = false
            Log.e(TAG, "WebRTC initialization failed", e)
            onStateChanged("FAILED")
        }
    }

    private fun ensureInitialized(): Boolean {
        if (!initialized) {
            initialize()
        }
        return initialized && factory != null && !closed
    }

    private fun createAudioTrack(): Boolean {
        if (audioTrack != null) {
            return true
        }

        val rtcFactory = factory ?: return false

        return try {
            val audioConstraints = MediaConstraints()
            audioSource = rtcFactory.createAudioSource(audioConstraints)
            audioTrack = rtcFactory.createAudioTrack(AUDIO_TRACK_ID, audioSource)
            audioTrack?.setEnabled(true)
            Log.d(TAG, "Local AudioTrack created and enabled")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Local AudioTrack creation failed", e)
            false
        }
    }

    private fun createPeerConnection(): Boolean {
        if (peerConnection != null) {
            return true
        }

        if (!ensureInitialized()) {
            Log.e(TAG, "Cannot create PeerConnection: WebRTC not initialized")
            return false
        }

        val rtcFactory = factory ?: return false

        return try {
            val iceServers = listOf(
                PeerConnection.IceServer.builder("stun:stun.l.google.com:19302")
                    .createIceServer()
            )

            val rtcConfig = PeerConnection.RTCConfiguration(iceServers).apply {
                sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
                continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
            }

            peerConnection = rtcFactory.createPeerConnection(
                rtcConfig,
                object : PeerConnection.Observer {

                    override fun onSignalingChange(state: PeerConnection.SignalingState) {
                        Log.d(TAG, "Signaling state changed: $state")
                    }

                    override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
                        Log.d(TAG, "ICE connection state: $state")
                        when (state) {
                            PeerConnection.IceConnectionState.CONNECTED,
                            PeerConnection.IceConnectionState.COMPLETED -> {
                                Log.d(TAG, "ICE connection established")
                                enableSpeakerphone()
                                onStateChanged("CONNECTED")
                            }
                            PeerConnection.IceConnectionState.FAILED -> {
                                Log.e(TAG, "ICE connection failed")
                                onStateChanged("FAILED")
                            }
                            PeerConnection.IceConnectionState.DISCONNECTED -> {
                                Log.w(TAG, "ICE connection disconnected")
                            }
                            else -> {}
                        }
                    }

                    override fun onConnectionChange(state: PeerConnection.PeerConnectionState) {
                        Log.d(TAG, "PeerConnection state changed: $state")
                        when (state) {
                            PeerConnection.PeerConnectionState.CONNECTED -> {
                                enableSpeakerphone()
                                onStateChanged("CONNECTED")
                            }
                            PeerConnection.PeerConnectionState.FAILED -> {
                                onStateChanged("FAILED")
                            }
                            PeerConnection.PeerConnectionState.CLOSED,
                            PeerConnection.PeerConnectionState.DISCONNECTED -> {
                                Log.d(TAG, "PeerConnection closed/disconnected")
                            }
                            else -> {}
                        }
                    }

                    override fun onIceCandidate(candidate: IceCandidate) {
                        Log.d(TAG, "Generated local ICE candidate: ${candidate.sdpMid}")
                        sendCandidate(candidate)
                    }

                    override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {}

                    override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
                        Log.d(TAG, "ICE gathering state: $state")
                    }

                    override fun onIceConnectionReceivingChange(receiving: Boolean) {}

                    override fun onAddStream(stream: org.webrtc.MediaStream) {
                        Log.d(TAG, "Remote MediaStream received with ${stream.audioTracks.size} audio tracks")
                        stream.audioTracks.forEach { it.setEnabled(true) }
                    }

                    override fun onRemoveStream(stream: org.webrtc.MediaStream) {
                        Log.d(TAG, "Remote MediaStream removed")
                    }

                    override fun onDataChannel(dataChannel: org.webrtc.DataChannel) {}

                    override fun onRenegotiationNeeded() {
                        Log.d(TAG, "WebRTC renegotiation requested")
                    }

                    override fun onAddTrack(receiver: RtpReceiver, mediaStreams: Array<out org.webrtc.MediaStream>) {
                        val track = receiver.track()
                        if (track is AudioTrack) {
                            track.setEnabled(true)
                            Log.d(TAG, "Remote AudioTrack received and enabled")
                        }
                    }
                }
            )

            val pc = peerConnection
            if (pc == null) {
                Log.e(TAG, "PeerConnection creation returned null")
                return false
            }

            if (!createAudioTrack()) {
                Log.e(TAG, "Failed to create audio track for PeerConnection")
                return false
            }

            val track = audioTrack ?: return false
            pc.addTrack(track, listOf(STREAM_ID))
            Log.d(TAG, "Local audio track added to PeerConnection (Stream: $STREAM_ID)")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Error creating PeerConnection", e)
            false
        }
    }

    /**
     * PHONE A: initiates call -> sends {"type": "call"}
     */
    fun startOutgoingCall() {
        if (!ensureInitialized()) {
            onStateChanged("FAILED")
            return
        }

        isCaller = true
        remoteDescriptionSet = false
        pendingCandidates.clear()

        signaling?.sendMessage(
            JSONObject()
                .put("type", "call")
                .toString()
        )

        onStateChanged("CALLING")
    }

    /**
     * PHONE B: user taps ACCEPT -> sends {"type": "call_accepted"}
     */
    fun acceptIncomingCall() {
        if (!ensureInitialized()) {
            onStateChanged("FAILED")
            return
        }

        isCaller = false
        remoteDescriptionSet = false
        pendingCandidates.clear()

        // Send call_accepted to tell Phone A it can create and send its WebRTC offer
        signaling?.sendMessage(
            JSONObject()
                .put("type", "call_accepted")
                .toString()
        )

        // Pre-create PeerConnection so it's ready to handle the incoming offer
        createPeerConnection()
        onStateChanged("CALLING")
    }

    /**
     * PHONE B: user taps REJECT -> sends {"type": "call_rejected"}
     */
    fun rejectIncomingCall() {
        signaling?.sendMessage(
            JSONObject()
                .put("type", "call_rejected")
                .toString()
        )

        closePeerConnection()
        onStateChanged("ENDED")
    }

    /**
     * PHONE A: creates WebRTC offer after receiving "call_accepted"
     */
    fun createOffer() {
        if (!isCaller) {
            Log.w(TAG, "createOffer called but not marked as caller")
            return
        }

        if (!createPeerConnection()) {
            onStateChanged("FAILED")
            return
        }

        val pc = peerConnection ?: return
        Log.d(TAG, "Creating SDP Offer...")

        val sdpConstraints = MediaConstraints().apply {
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "true"))
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "false"))
        }

        pc.createOffer(
            object : org.webrtc.SdpObserver {
                override fun onCreateSuccess(desc: SessionDescription) {
                    Log.d(TAG, "Offer creation succeeded, setting local description")
                    pc.setLocalDescription(
                        object : org.webrtc.SdpObserver {
                            override fun onSetSuccess() {
                                Log.d(TAG, "Local offer set successfully, sending to peer")
                                val msg = JSONObject()
                                    .put("type", "offer")
                                    .put("sdp", desc.description)
                                signaling?.sendMessage(msg.toString())
                            }

                            override fun onSetFailure(error: String) {
                                Log.e(TAG, "Failed to set local offer: $error")
                                onStateChanged("FAILED")
                            }

                            override fun onCreateSuccess(d: SessionDescription) {}
                            override fun onCreateFailure(e: String) {}
                        },
                        desc
                    )
                }

                override fun onCreateFailure(error: String) {
                    Log.e(TAG, "Offer creation failed: $error")
                    onStateChanged("FAILED")
                }

                override fun onSetSuccess() {}
                override fun onSetFailure(error: String) {}
            },
            sdpConstraints
        )
    }

    /**
     * PHONE B: handles incoming offer -> sets remote description -> creates answer
     */
    private fun handleRemoteOffer(sdp: String) {
        if (!createPeerConnection()) {
            onStateChanged("FAILED")
            return
        }

        val pc = peerConnection ?: return
        val offer = SessionDescription(SessionDescription.Type.OFFER, sdp)

        Log.d(TAG, "Setting remote offer description...")
        pc.setRemoteDescription(
            object : org.webrtc.SdpObserver {
                override fun onSetSuccess() {
                    Log.d(TAG, "Remote offer set successfully. Creating answer...")
                    remoteDescriptionSet = true
                    flushPendingCandidates()
                    createAnswer()
                }

                override fun onSetFailure(error: String) {
                    Log.e(TAG, "Failed to set remote offer: $error")
                    onStateChanged("FAILED")
                }

                override fun onCreateSuccess(d: SessionDescription) {}
                override fun onCreateFailure(e: String) {}
            },
            offer
        )
    }

    /**
     * PHONE B: creates WebRTC answer
     */
    private fun createAnswer() {
        val pc = peerConnection ?: return

        val sdpConstraints = MediaConstraints().apply {
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "true"))
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "false"))
        }

        Log.d(TAG, "Creating SDP Answer...")
        pc.createAnswer(
            object : org.webrtc.SdpObserver {
                override fun onCreateSuccess(desc: SessionDescription) {
                    Log.d(TAG, "Answer creation succeeded, setting local description")
                    pc.setLocalDescription(
                        object : org.webrtc.SdpObserver {
                            override fun onSetSuccess() {
                                Log.d(TAG, "Local answer set successfully, sending to peer")
                                val msg = JSONObject()
                                    .put("type", "answer")
                                    .put("sdp", desc.description)
                                signaling?.sendMessage(msg.toString())
                            }

                            override fun onSetFailure(error: String) {
                                Log.e(TAG, "Failed to set local answer: $error")
                                onStateChanged("FAILED")
                            }

                            override fun onCreateSuccess(d: SessionDescription) {}
                            override fun onCreateFailure(e: String) {}
                        },
                        desc
                    )
                }

                override fun onCreateFailure(error: String) {
                    Log.e(TAG, "Answer creation failed: $error")
                    onStateChanged("FAILED")
                }

                override fun onSetSuccess() {}
                override fun onSetFailure(error: String) {}
            },
            sdpConstraints
        )
    }

    /**
     * PHONE A: handles incoming answer -> sets remote description
     */
    private fun handleRemoteAnswer(sdp: String) {
        val pc = peerConnection
        if (pc == null) {
            Log.e(TAG, "PeerConnection is null when receiving answer")
            return
        }

        val answer = SessionDescription(SessionDescription.Type.ANSWER, sdp)
        Log.d(TAG, "Setting remote answer description...")

        pc.setRemoteDescription(
            object : org.webrtc.SdpObserver {
                override fun onSetSuccess() {
                    Log.d(TAG, "Remote answer set successfully")
                    remoteDescriptionSet = true
                    flushPendingCandidates()
                }

                override fun onSetFailure(error: String) {
                    Log.e(TAG, "Failed to set remote answer: $error")
                    onStateChanged("FAILED")
                }

                override fun onCreateSuccess(d: SessionDescription) {}
                override fun onCreateFailure(e: String) {}
            },
            answer
        )
    }

    /**
     * Process all signaling messages from WebSocket
     */
    fun handleSignalingMessage(message: String) {
        try {
            val json = JSONObject(message)
            when (val type = json.optString("type")) {
                "offer" -> {
                    val sdp = json.optString("sdp")
                    if (sdp.isNotEmpty()) {
                        handleRemoteOffer(sdp)
                    }
                }
                "answer" -> {
                    val sdp = json.optString("sdp")
                    if (sdp.isNotEmpty()) {
                        handleRemoteAnswer(sdp)
                    }
                }
                "candidate" -> {
                    val sdp = json.optString("candidate")
                    if (sdp.isNotEmpty()) {
                        val sdpMid = if (json.has("sdpMid")) json.getString("sdpMid") else ""
                        val sdpMLineIndex = json.optInt("sdpMLineIndex", 0)
                        val candidate = IceCandidate(sdpMid, sdpMLineIndex, sdp)

                        if (!remoteDescriptionSet || peerConnection == null) {
                            Log.d(TAG, "Queuing remote ICE candidate")
                            pendingCandidates.add(candidate)
                        } else {
                            Log.d(TAG, "Adding remote ICE candidate directly")
                            peerConnection?.addIceCandidate(candidate)
                        }
                    }
                }
                else -> {
                    Log.d(TAG, "Signaling message of type '$type' handled by Activity")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling signaling message: $message", e)
        }
    }

    private fun sendCandidate(candidate: IceCandidate) {
        try {
            val msg = JSONObject()
                .put("type", "candidate")
                .put("candidate", candidate.sdp)
                .put("sdpMid", candidate.sdpMid)
                .put("sdpMLineIndex", candidate.sdpMLineIndex)

            signaling?.sendMessage(msg.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to serialize/send ICE candidate", e)
        }
    }

    private fun flushPendingCandidates() {
        val pc = peerConnection ?: return
        if (pendingCandidates.isEmpty()) return

        Log.d(TAG, "Flushing ${pendingCandidates.size} pending ICE candidates")
        val copy = pendingCandidates.toList()
        pendingCandidates.clear()
        copy.forEach { pc.addIceCandidate(it) }
    }

    /**
     * Terminate call: sends hangup, disposes PeerConnection, resets call state
     */
    fun endCall() {
        Log.d(TAG, "Ending call...")
        try {
            signaling?.sendMessage(
                JSONObject().put("type", "hangup").toString()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error sending hangup", e)
        }

        closePeerConnection()
        isCaller = false
        onStateChanged("ENDED")
    }

    @Suppress("DEPRECATION")
    private fun enableSpeakerphone() {
        try {
            audioManager?.mode = android.media.AudioManager.MODE_IN_COMMUNICATION
            audioManager?.isSpeakerphoneOn = true
            Log.d(TAG, "Speakerphone enabled (MODE_IN_COMMUNICATION)")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to enable speakerphone: ${e.message}")
        }
    }

    @Suppress("DEPRECATION")
    private fun resetAudioMode() {
        try {
            audioManager?.mode = android.media.AudioManager.MODE_NORMAL
            audioManager?.isSpeakerphoneOn = false
            Log.d(TAG, "Audio mode reset to MODE_NORMAL")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to reset audio mode: ${e.message}")
        }
    }

    fun closePeerConnection() {
        resetAudioMode()
        remoteDescriptionSet = false
        pendingCandidates.clear()

        try {
            peerConnection?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing peerConnection: ${e.message}")
        }

        try {
            peerConnection?.dispose()
        } catch (e: Exception) {
            Log.w(TAG, "Error disposing peerConnection: ${e.message}")
        }
        peerConnection = null

        try {
            audioTrack?.setEnabled(false)
            audioTrack?.dispose()
        } catch (e: Exception) {
            Log.w(TAG, "Error disposing audioTrack: ${e.message}")
        }
        audioTrack = null

        try {
            audioSource?.dispose()
        } catch (e: Exception) {
            Log.w(TAG, "Error disposing audioSource: ${e.message}")
        }
        audioSource = null
    }

    /**
     * Complete lifecycle release (Activity onDestroy)
     */
    fun close() {
        if (closed) return
        closed = true
        Log.d(TAG, "Releasing all WebRTC resources...")

        closePeerConnection()

        try {
            audioDeviceModule?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing audioDeviceModule: ${e.message}")
        }
        audioDeviceModule = null

        try {
            factory?.dispose()
        } catch (e: Exception) {
            Log.w(TAG, "Error disposing PeerConnectionFactory: ${e.message}")
        }
        factory = null

        initialized = false
        Log.d(TAG, "WebRTC resources fully released")
    }
}