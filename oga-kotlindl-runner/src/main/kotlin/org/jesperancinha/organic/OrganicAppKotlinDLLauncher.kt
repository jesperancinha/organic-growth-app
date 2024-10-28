package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.dataset.evaluate
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO


class OrganicAppKotlinDLLauncher

fun main() {
    val (train, test) = mnist()
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
            loss = Losses.MSE,
            metric = Metrics.ACCURACY
        )
        it.fit(dataset = train, epochs = 10, batchSize = 32)

        val accuracy = it.evaluate(dataset =  test).metrics[Metrics.ACCURACY]

        println("Test accuracy: $accuracy")

        val modelDirectory = File("mnist_model")
        modelDirectory.exists().takeIf { false } ?: modelDirectory.deleteRecursively()

        it.save(modelDirectory, SavingFormat.TF_GRAPH_CUSTOM_VARIABLES)
    }
    TensorFlowInferenceModel.load(File("mnist_model")).use {
        it.reshape(28, 28, 1)
        it.predictImage("one.png")
        it.predictImage("three.png")
        it.predictImage("four.png")
        val accuracy = it.evaluate(
            dataset = test,
            metric = Metrics.ACCURACY
        )
        println("Test accuracy: $accuracy")
    }

}

fun TensorFlowInferenceModel.predictImage(
    imageFile: String
) {
    val resourceAsStream = OrganicAppKotlinDLLauncher::class.java.getResourceAsStream("/$imageFile")
    val preprocessedImage =
        ImageConverter.toNormalizedFloatArray(ImageConverter.toBufferedImage(resourceAsStream), ColorMode.GRAYSCALE)
    val prediction = predict(preprocessedImage)
    println("Predicted class: ${prediction}")
}

fun FloatArray.saveFloatArrayAsImage(
    width: Int, height: Int, outputFile: File
) {
    require(this.size == width * height) { "Array size must be width * height" }

    val image = BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)

    for (y in 0 until height) {
        for (x in 0 until width) {
            val normalizedValue = (this[y * width + x] * 255).toInt().coerceIn(0, 255)
            val rgb = (normalizedValue shl 16) or (normalizedValue shl 8) or normalizedValue
            image.setRGB(x, y, rgb)
        }
    }
    ImageIO.write(image, "png", outputFile)
}