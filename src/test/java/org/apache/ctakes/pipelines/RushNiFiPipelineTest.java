package org.apache.ctakes.pipelines;

import org.apache.commons.io.FileUtils;
import org.apache.ctakes.utils.RushConfig;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import static org.junit.Assert.assertEquals;


public class RushNiFiPipelineTest {
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

    @SuppressWarnings("unused")
    @Test
    public void testExecutePipeline() throws Exception {

        File inputDirectory = Paths.get("src/test/resources/input").toFile();
        File expectedXMIsDirectory = Paths.get("src/test/resources/expectedOutput/xmis/").toFile();
        File expectedCUIsDirectory = Paths.get("src/test/resources/expectedOutput/cuis/").toFile();
        File expectedGranularDirectory = Paths.get("src/test/resources/expectedOutput/granular/").toFile();

        File masterFolder = Paths.get("resources").toFile();
        File tempMasterFolder = folder.newFolder("tempMasterFolder");
        File outputDirectory = folder.newFolder("outputDirectory");
        File actualCuis = new File(outputDirectory, "cuis");
        File actualXmis = new File(outputDirectory, "xmis");
        File actualGranular = new File(outputDirectory, "granular");

        try (RushConfig config = new RushConfig(masterFolder.getAbsolutePath(), tempMasterFolder.getAbsolutePath())) {
            config.initialize();
            try (RushNiFiPipeline pipeline = new RushNiFiPipeline(config, true)) {
                pipeline.execute(inputDirectory, outputDirectory);
            }
        }

        for (File file : Objects.requireNonNull(inputDirectory.listFiles())) {

            // TODO find way to compare directly
//            String xmi = FileUtils.readFileToString(new File(actualXmis, file.getName()));
//            String expectedXmi = FileUtils.readFileToString(new File(expectedXMIsDirectory, file.getName()));
//            assertEquals(xmi, expectedXmi);

            String cuis = FileUtils.readFileToString(new File(actualCuis, file.getName()));
            String expectedCuis = FileUtils.readFileToString(new File(expectedCUIsDirectory, file.getName()));
            assertEquals(expectedCuis, cuis);

            String granular = FileUtils.readFileToString(new File(actualGranular, file.getName()));
            String expectedGranular = FileUtils.readFileToString(new File(expectedGranularDirectory, file.getName()));
            assertEquals(expectedGranular, granular);
        }
    }
}