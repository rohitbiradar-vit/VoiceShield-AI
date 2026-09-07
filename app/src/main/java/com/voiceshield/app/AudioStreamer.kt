package com.voiceshield.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString

class AudioStreamer(
    private val context: Context,
    private val serverUrl: String
) {

    companion object {

        private const val TAG = "AudioStreamer"

        private const val SAMPLE_RATE = 16000

        private const val CHANNEL_CONFIG =
            AudioFormat.CHANNEL_IN_MONO

        private const val AUDIO_FORMAT =
            AudioFormat.ENCODING_PCM_16BIT
    }

    private var audioRecord: AudioRecord? = null

    private var webSocket: WebSocket? = null

    private var recordingThread: Thread? = null

    @Volatile
    private var isStreaming = false

    fun start() {

        if (
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            Log.e(
                TAG,
                "RECORD_AUDIO permission not granted"
            )

            return
        }

        if (isStreaming) {
            return
        }

        val minBufferSize =
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT
            )

        if (minBufferSize <= 0) {

            Log.e(
                TAG,
                "Invalid AudioRecord buffer size"
            )

            return
        }

        val bufferSize =
            maxOf(
                minBufferSize * 2,
                2048
            )

        try {

            audioRecord =
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    bufferSize
                )

            if (
                audioRecord?.state !=
                AudioRecord.STATE_INITIALIZED
            ) {

                Log.e(
                    TAG,
                    "AudioRecord initialization failed"
                )

                stop()
                return
            }

            val request =
                Request.Builder()
                    .url(serverUrl)
                    .build()

            val client =
                OkHttpClient()

            webSocket =
                client.newWebSocket(
                    request,
                    object : WebSocketListener() {

                        override fun onOpen(
                            webSocket: WebSocket,
                            response: okhttp3.Response
                        ) {

                            Log.d(
                                TAG,
                                "Audio WebSocket connected"
                            )
                        }

                        override fun onFailure(
                            webSocket: WebSocket,
                            t: Throwable,
                            response: okhttp3.Response?
                        ) {

                            Log.e(
                                TAG,
                                "Audio WebSocket failure",
                                t
                            )
                        }

                        override fun onClosed(
                            webSocket: WebSocket,
                            code: Int,
                            reason: String
                        ) {

                            Log.d(
                                TAG,
                                "Audio WebSocket closed"
                            )
                        }
                    }
                )

            isStreaming = true

            audioRecord?.startRecording()

            recordingThread =
                Thread {

                    val buffer =
                        ByteArray(bufferSize)

                    while (isStreaming) {

                        val recorder =
                            audioRecord
                                ?: break

                        val bytesRead =
                            try {

                                recorder.read(
                                    buffer,
                                    0,
                                    buffer.size
                                )

                            } catch (e: Exception) {

                                Log.e(
                                    TAG,
                                    "Audio read failed",
                                    e
                                )

                                break
                            }

                        if (bytesRead > 0) {

                            val data =
                                buffer.toByteString(
                                    0,
                                    bytesRead
                                )

                            webSocket?.send(
                                data
                            )
                        }
                    }
                }

            recordingThread?.start()

        } catch (e: Exception) {

            Log.e(
                TAG,
                "AudioStreamer start failed",
                e
            )

            stop()
        }
    }

    fun stop() {

        isStreaming = false

        try {
            recordingThread?.interrupt()
        } catch (_: Exception) {
        }

        try {
            audioRecord?.stop()
        } catch (_: Exception) {
        }

        try {
            audioRecord?.release()
        } catch (_: Exception) {
        }

        audioRecord = null

        try {
            webSocket?.close(
                1000,
                "Audio stopped"
            )
        } catch (_: Exception) {
        }

        webSocket = null
        recordingThread = null

        Log.d(
            TAG,
            "AudioStreamer stopped"
        )
    }
}