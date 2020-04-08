package org.apache.ctakes.pipelines;

import org.apache.commons.io.FileUtils;
import org.apache.ctakes.utils.RushConfig;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;


public class RushEndToEndPipelineTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void name() {
    }

    @Test
    public void testPipeline() throws Exception {

        FileUtils.forceMkdir(new File("/tmp/random")); // required for current implementation...

        File inputDirectory = Paths.get("src/test/resources/input").toFile();
        File outputDirectory = Paths.get("src/test/resources/expectedOutput").toFile();

        File masterFolder = Paths.get("resources").toFile();
        File tempMasterFolder = folder.newFolder("tempMasterFolder");

        RushConfig config = new RushConfig(masterFolder.getAbsolutePath(),
                tempMasterFolder.getAbsolutePath());
        config.initialize();
        RushEndToEndPipeline pipeline = new RushEndToEndPipeline(config, true);

        for (File file : inputDirectory.listFiles()) {
            String t = FileUtils.readFileToString(file);
            CTakesResult result = pipeline.getResult(file.getAbsolutePath(), 1, t);

            String expectedOutput = FileUtils.readFileToString(new File(outputDirectory.getAbsolutePath() + "/xmis/" + file.getName()));
            String expectedCuis = FileUtils.readFileToString(new File(outputDirectory.getAbsolutePath() + "/cuis/" + file.getName()));

//            assertEquals(expectedOutput,result.getOutput()); //TODO find way to compare
            assertEquals(expectedCuis, result.getCuis());
        }
        System.out.println("Closing Pipeline");
        pipeline.close();
        config.close();
    }

}