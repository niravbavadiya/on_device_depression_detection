package com.example.depressiondetectionstandalone.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface SensorDao {
    // Location data access
    @Insert
    suspend fun insertLocation(e: LocationEntity)

    @Query("SELECT * FROM location")
    suspend fun getAllLocations(): List<LocationEntity>

    // Screen data access
    @Insert
    suspend fun insertScreen(e: ScreenEntity)

    @Query("SELECT * FROM screen")
    suspend fun getAllScreens(): List<ScreenEntity>

    // Steps data access
    @Insert
    suspend fun insertSteps(e: StepsEntity)

    @Query("SELECT * FROM steps")
    suspend fun getAllSteps(): List<StepsEntity>

    // Sleep data access
    @Insert
    suspend fun insertSleep(e: SleepEntity)

    @Query("SELECT * FROM sleep")
    suspend fun getAllSleep(): List<SleepEntity>
}
