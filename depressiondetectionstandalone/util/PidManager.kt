package com.example.depressiondetectionstandalone.util

import android.content.Context
import android.provider.Settings
import java.util.UUID

object PidManager {
    private const val PREF_NAME = "device_id_prefs"
    private const val KEY_PID = "pid"

    /** Returns the persistent per-device ID. Generates and stores it if not already set. */
    fun getOrCreatePid(context: Context): String {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        // If an ID was already generated and stored, return it
        prefs.getString(KEY_PID, null)?.let { return it }

        // No ID yet: generate a new one
        val newId: String = Settings.Secure.getString(
            context.contentResolver, Settings.Secure.ANDROID_ID
        ) ?: UUID.randomUUID().toString()
        // Store the ID for future use
        prefs.edit().putString(KEY_PID, newId).apply()
        return newId
    }
}
