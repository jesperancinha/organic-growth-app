package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics.ACCURACY
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.junit.jupiter.api.Test
import java.io.File

class OrganicAppKotlinDLCheckpointTest {

    @Test
    fun `should test number recognition application`() {
        val (train, test) = mnist()
        val model = Sequential.of(
            Input(28, 28, 1),
            Flatten(),
            Dense(128, Activations.Relu),
            Dense(64, Activations.Relu),
            Dense(10, Activations.Softmax)
        )

        model.use {
            it.compile(
                optimizer = Adam(),
                loss = SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = ACCURACY
            )
            var bestAccuracy = 0.0
            for (epoch in 1..10) {
                println("Epoch $epoch")
                it.fit(train, epochs = 1, batchSize = 32)
                val metrics = it.evaluate(test)
                val accuracy = metrics.metrics[ACCURACY] ?: 0.0
                println("Validation accuracy: $accuracy")
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy
                    val modelDirectory = File("saved_model")
                    modelDirectory.exists().takeIf { false } ?: modelDirectory.deleteRecursively()
                    modelDirectory.mkdirs()

                    val saveFile = File("saved_model/best_model.zip")
                    it.saveModelConfiguration(saveFile)
                    it.save(saveFile, writingMode = WritingMode.OVERRIDE)
                    println("Model improved. Saved at epoch $epoch with accuracy $accuracy")
                }
            }
        }
    }

}