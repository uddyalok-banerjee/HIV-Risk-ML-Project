package org.apache.ctakes.consumers;

import org.apache.commons.io.FileUtils;
import org.apache.ctakes.pipelines.RushSimplePipeline;
import org.apache.ctakes.utils.Utils;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.*;

public class GranularJsonWriterTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @Before
    public void before() throws IOException {
        // make sure setup is correct, including a couple "hardcoded" paths
        FileUtils.forceMkdir(new File("/tmp/random")); // required for current implementation...

        Path link = Paths.get("/tmp/ctakes-config");
        if (Files.exists(link)) {
            Files.delete(link);
        }
        Files.createSymbolicLink(link, Paths.get("resources").toAbsolutePath());
    }

    @Test
    public void testEngine() throws Exception {

        File inputDirectory = Paths.get("src/test/resources/input").toFile();
        File outputDirectory = Paths.get("src/test/resources/expectedOutput").toFile();
        File expectedXMIsDirectory = Paths.get("src/test/resources/expectedOutput/xmis/").toFile();
        File expectedOverviewsDirectory = Paths.get("src/test/resources/expectedOutput/overviews/").toFile();

        AnalysisEngine engine = AnalysisEngineFactory.createEngine(GranularJsonWriter.class);

        for (File file : expectedXMIsDirectory.listFiles()) {
            String xmi = FileUtils.readFileToString(file);
            CollectionReader xmlCollectionReader = Utils.getCollectionReader(xmi);

            String overview = RushSimplePipeline.runPipeline(xmlCollectionReader, engine);

            String expectedOverview = FileUtils.readFileToString(new File(expectedOverviewsDirectory, file.getName()));

            assertEquals(expectedOverview, overview);
        }
    }

}