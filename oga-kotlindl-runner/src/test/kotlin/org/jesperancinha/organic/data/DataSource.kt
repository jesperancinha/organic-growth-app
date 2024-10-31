package org.jesperancinha.organic.data

import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import kotlin.random.Random

fun generateSyntheticData(): Pair<OnHeapDataset, OnHeapDataset> {
    val random = Random(42)

    val xTrainRaw = (0..100)
        .map { it.toFloat() }
        .toFloatArray()
    val yTrainRaw = (0..100)
        .map { (2 * it + 1 + random.nextDouble() * 10).toFloat() }
        .toFloatArray()

    val xTestRaw = (101..200)
        .map { it.toFloat() }.toFloatArray()
    val yTestRaw = (101..200)
        .map { (2 * it + 1 + random.nextDouble() * 10).toFloat() }
        .toFloatArray()

    val xTrain = xTrainRaw
        .map { floatArrayOf(it / 200f) }
        .toTypedArray()
    val yTrain = yTrainRaw
        .map { it / 200f }
        .toFloatArray()

    val xTest = xTestRaw
        .map { floatArrayOf(it / 200f) }
        .toTypedArray()
    val yTest = yTestRaw
        .map { it / 200f }
        .toFloatArray()

    val trainData = OnHeapDataset
        .create(xTrain, yTrain)
    val testData = OnHeapDataset
        .create(xTest, yTest)
    return Pair(trainData, testData)
}