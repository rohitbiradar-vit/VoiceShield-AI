package com.voiceshield.app

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.util.concurrent.TimeUnit

class WebSocketManager(
    private val serverUrl: String = "ws://192.168.0.106:8000/ws/room1",
    private val onMessage: (String) -> Unit,
    private val onConnectionState: ((String) -> Unit)? = null
) {

    companion object {
        private const val TAG = "VoiceShieldWS"
    }

    private var webSocket: WebSocket? = null

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(10, TimeUnit.SECONDS)
        .build()

    @Volatile
    private var connected = false

    fun isConnected(): Boolean = connected

    fun connect() {
        if (connected && webSocket != null) {
            Log.d(TAG, "Already connected to $serverUrl")
            return
        }

        Log.d(TAG, "Connecting to $serverUrl...")

        val request = Request.Builder()
            .url(serverUrl)
            .build()

        webSocket = client.newWebSocket(
            request,
            object : WebSocketListener() {

                override fun onOpen(webSocket: WebSocket, response: Response) {
                    Log.d(TAG, "WebSocket CONNECTED to $serverUrl")
                    connected = true
                    onConnectionState?.invoke("CONNECTED")
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    Log.d(TAG, "RECEIVED: $text")
                    onMessage(text)
                }

                override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                    Log.d(TAG, "RECEIVED BYTES: ${bytes.size} bytes")
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    Log.e(TAG, "WebSocket ERROR: ${t.message}", t)
                    connected = false
                    onConnectionState?.invoke("FAILED")
                }

                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    Log.d(TAG, "WebSocket CLOSING: $code / $reason")
                    connected = false
                    onConnectionState?.invoke("DISCONNECTED")
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    Log.d(TAG, "WebSocket CLOSED: $code / $reason")
                    connected = false
                    onConnectionState?.invoke("DISCONNECTED")
                }
            }
        )
    }

    fun sendMessage(message: String) {
        Log.d(TAG, "SEND: $message")
        val sent = webSocket?.send(message) ?: false
        if (!sent) {
            Log.w(TAG, "Failed to send message (socket not open): $message")
        }
    }

    fun sendBytes(data: ByteArray, offset: Int = 0, byteCount: Int = data.size) {
        val byteString = data.toByteString(offset, byteCount)
        webSocket?.send(byteString)
    }

    fun disconnect() {
        Log.d(TAG, "Disconnecting WebSocket...")
        connected = false
        try {
            webSocket?.close(1000, "VoiceShield clean close")
        } catch (e: Exception) {
            Log.w(TAG, "Exception during WebSocket close: ${e.message}")
        }
        webSocket = null
    }
}