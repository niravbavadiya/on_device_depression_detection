package com.example.depressiondetectionstandalone.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "screen")
data class ScreenEntity(
    @PrimaryKey(autoGenerate = true) val localId: Long = 0,
    val pid: String,
    val timestamp: Long,
    val screen_status: String,
    val brightness: Int?,
    val interactive: Boolean?,
    val orientation: Int?
)
