package org.jesperancinha.oga;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;

@Disabled
public class OgaDeepLearning4JRunnerTest {

    @Test
    public void testOgaDeepLearning4JRunner() throws IOException {
        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;
        int batchSize = 32;
        File mainPath = new File("fashion-images");
        if (!mainPath.exists()) {
            throw new RuntimeException("Could not find fashion-images");
        }
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random());
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(fileSplit);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            try {
                ds.getFeatures().reshape(batchSize, height * width * channels);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width * channels)
                        .nOut(128)
                        .activation(RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(outputNum)
                        .activation(SOFTMAX)
                        .build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        int numEpochs = 10;
        for (int i = 0; i < numEpochs; i++) {
            model.fit(dataIter);
            dataIter.reset();
            System.out.println("Epoch " + (i + 1) + " completed.");
        }

        System.out.println("Model training completed.");
    }
}
