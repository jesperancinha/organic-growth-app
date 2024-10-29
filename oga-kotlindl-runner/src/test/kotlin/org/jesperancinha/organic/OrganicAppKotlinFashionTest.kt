package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.junit.jupiter.api.Test
import java.io.File

const val FASHION_FOLDER = "fashion-images"

class OrganicAppKotlinFashionTest {

    val stringLabels = mapOf(
        0 to "T-shirt/top",
        1 to "Trousers",
        2 to "Pullover",
        3 to "Dress",
        4 to "Coat",
        5 to "Sandals",
        6 to "Shirt",
        7 to "Sneakers",
        8 to "Bag",
        9 to "Ankle boots"
    )

    @Test
    fun `should make preliminary tests to fashion`() {
        val (train, test) = fashionMnist()

        val model = Sequential.of(
            Input(28, 28, 1),
            Flatten(),
            Dense(300),
            Dense(100),
            Dense(10)
        )

        model.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.printSummary()

            it.fit(
                dataset = train,
                epochs = 10,
                batchSize = 100
            )

            val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            println("Accuracy: $accuracy")

            val modelDirectory = File("fashion-mnist")
            modelDirectory.exists().takeIf { false } ?: modelDirectory.deleteRecursively()

            it.save(File("fashion-mnist"), writingMode = WritingMode.OVERRIDE)
        }

        TensorFlowInferenceModel.load(File("fashion-mnist")).use {
            it.reshape(28, 28, 1)
            val image1 = test.getX(0)
            val prediction = it.predict(image1)
            val actualLabel = test.getY(0)

            println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
            println("Actual label is: $actualLabel.")
        }
    }


    @Test
    fun `should generate mnist image sets`() {
        val (train, test) = fashionMnist()
        File(FASHION_FOLDER).mkdirs()
        test.x.forEachIndexed { i, image ->
            val pathname = stringLabels[test.getY(i).toInt()]
            File("$FASHION_FOLDER/$pathname").mkdirs()
            image.saveFloatArrayAsImage(28, 28, File("fashion-images/$pathname/fashion$i.png"))

        }
    }
}