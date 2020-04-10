package org.apache.ctakes.pipelines;

import com.google.gson.Gson;
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
    private static Gson gson = new Gson();

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
    @Test(timeout = 60_000)
    public void testExecutePipeline() throws Exception {

        File inputDirectory = Paths.get("src/test/resources/input").toFile();
        File outputDirectory = folder.newFolder("outputDirectory");
        File masterFolder = Paths.get("resources").toFile();
        File tempMasterFolder = folder.newFolder("tempMasterFolder");

        // execute pipeline
        try (RushConfig config = new RushConfig(masterFolder.getAbsolutePath(), tempMasterFolder.getAbsolutePath())) {
            config.initialize();
            try (RushNiFiPipeline pipeline = new RushNiFiPipeline(config, true) {

                void writeXmi(File outputDirectory, String fileName, String xmi) throws IOException {
                    write(outputDirectory, "xmis", fileName, xmi);
                }

                void writeCui(File outputDirectory, String fileName, String xmi) throws Exception {
                    write(outputDirectory, "cuis", fileName, getCuis(xmi));
                }
            }) {
                pipeline.execute(inputDirectory, outputDirectory);
            }
        }

        //check results

        final String expectedOutputDir = "src/test/resources/expectedOutput";

        File actualCuiDir = new File(outputDirectory, "cuis");
        File actualXmiDir = new File(outputDirectory, "xmis");
        File actualGranularDir = new File(outputDirectory, "granular");
        File actualOverviewDir = new File(outputDirectory, "overview");

        for (File file : Objects.requireNonNull(inputDirectory.listFiles())) {

            // TODO find way to compare directly
            String actualXmi = FileUtils.readFileToString(new File(actualXmiDir, file.getName()));
//            String expectedXmi = FileUtils.readFileToString(new File(expectedXMIsDirectory, file.getName()));
//            assertEquals(xmi, expectedXmi);

            String actualCuis = FileUtils.readFileToString(new File(actualCuiDir, file.getName()));
            String expectedCuis = FileUtils.readFileToString(Paths.get(expectedOutputDir, "cuis", file.getName()).toFile());
            assertEquals(expectedCuis, actualCuis);

            String actualGranular = FileUtils.readFileToString(new File(actualGranularDir, file.getName()));
            String expectedGranular = FileUtils.readFileToString(Paths.get(expectedOutputDir, "granular", file.getName()).toFile());
            assertEquals(file.getName(), expectedGranular, actualGranular);

            String actualOverview = FileUtils.readFileToString(new File(actualOverviewDir, file.getName()));

            RushNiFiPipeline.Overview overview = gson.fromJson(actualOverview, RushNiFiPipeline.Overview.class);
            assertEquals("", overview.fname);
            assertEquals("", overview.loadID);
            assertEquals("", overview.loadTimestamp);

            assertEquals(actualXmi, overview.xmi);

            String inputText = FileUtils.readFileToString(new File(inputDirectory, file.getName()));
            assertEquals(inputText, overview.rawText);
        }
    }
}