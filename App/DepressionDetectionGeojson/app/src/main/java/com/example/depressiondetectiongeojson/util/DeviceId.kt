package com.example.depressiondetectiongeojson.util

import android.content.Context
import android.provider.Settings

object DeviceId {
    fun pid(context: Context): String =
        Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID) ?: "device"
}
