package com.example.depressiondetectionstandalone.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [LocationEntity::class, ScreenEntity::class, StepsEntity::class, SleepEntity::class],
    version = 2,                   // Version updated for schema change (pid added)
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    // Access to the DAO
    abstract fun dao(): SensorDao

    companion object {
        @Volatile private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            // Singleton pattern to get the database instance
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "aware_like_db"                // Database file name
                )
                    .fallbackToDestructiveMigration() // Wipes and rebuilds if no migration (fresh start)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
