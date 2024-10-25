package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Linear
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Relu
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.Losses.MSE
import org.jetbrains.kotlinx.dl.api.core.loss.MAE
import org.jetbrains.kotlinx.dl.api.core.loss.MSE
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics.MAE
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.jupiter.api.Test
import kotlin.random.Random

class OrganicAppUnderfittingCorrectionTest {

    @Test
    fun `should run a model created with corrected underfitting`() {
        val (trainData, testData) = generateNormalizedSyntheticData()

        val model = Sequential.of(
            Input(1),
            Dense(128, activation = Relu),
            Dense(64, activation = Relu),
            Dense(32, activation = Relu),
            Dense(1, activation = Linear)
        )

        model.compile(
            optimizer = Adam(learningRate = 0.01f),
            loss = MSE,
            metric = MAE
        )

        model.fit(trainData, epochs = 1000, batchSize = 32)

        val evaluation = model.evaluate(testData)
        println("Test Loss: ${evaluation.lossValue}, Test MAE: ${evaluation.metrics[MAE]}")
    }

    fun generateNormalizedSyntheticData(): Pair<OnHeapDataset, OnHeapDataset> {
        val random = Random(42)
        val xTrainRaw = (0..100).map { it.toFloat() }.toFloatArray()
        val yTrainRaw = (0..100).map { (2 * it + 1 + random.nextDouble() * 10).toFloat() }.toFloatArray()
        val xTestRaw = (101..200).map { it.toFloat() }.toFloatArray()
        val yTestRaw = (101..200).map { (2 * it + 1 + random.nextDouble() * 10).toFloat() }.toFloatArray()
        val xTrain = xTrainRaw.map { floatArrayOf(it / 200f) }.toTypedArray()
        val yTrain = yTrainRaw.map { it / 200f }.toFloatArray()
        val xTest = xTestRaw.map { floatArrayOf(it / 200f) }.toTypedArray()
        val yTest = yTestRaw.map { it / 200f }.toFloatArray()
        val trainData = OnHeapDataset.create(xTrain, yTrain)
        val testData = OnHeapDataset.create(xTest, yTest)
        return Pair(trainData, testData)
    }
}