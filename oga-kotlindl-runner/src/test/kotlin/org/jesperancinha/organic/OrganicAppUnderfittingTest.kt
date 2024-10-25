package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.MSE
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.jupiter.api.Test
import kotlin.random.Random

class OrganicAppUnderfittingTest {

    @Test
    fun `should run a model created with underfitting`() {
        val (trainData, testData) = generateSyntheticData()
        val model = Sequential.of(
            Input(1),
            Dense(64, activation = Activations.Relu),
            Dense(1, activation = Activations.Linear)
        )
        model.compile(
            optimizer = Adam(),
            loss = Losses.MSE,
            metric = Metrics.MAE
        )
        model.fit(trainData, epochs = 100, batchSize = 32)
        val evaluation = model.evaluate(testData)
        println("Test Loss: ${evaluation.lossValue}, Test MAE: ${evaluation.metrics[Metrics.MAE]}")
    }

    fun generateSyntheticData(): Pair<OnHeapDataset, OnHeapDataset> {
        val random = Random(42)
        val xTrain = Array(101) { floatArrayOf(it.toFloat()) }
        val yTrain = FloatArray(101) { (2 * it + 1 + random.nextDouble() * 10).toFloat() }
        val xTest = Array(100) { floatArrayOf((101 + it).toFloat()) }
        val yTest = FloatArray(100) { (2 * (101 + it) + 1 + random.nextDouble() * 10).toFloat() }
        val trainData = OnHeapDataset.create(xTrain, yTrain)
        val testData = OnHeapDataset.create(xTest, yTest)
        return Pair(trainData, testData)
    }

}