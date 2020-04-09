package org.apache.ctakes.pipelines;

import com.google.common.base.Throwables;
import com.google.gson.Gson;
import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;
import org.apache.commons.io.FileUtils;
import org.apache.ctakes.assertion.medfacts.cleartk.PolarityCleartkAnalysisEngine;
import org.apache.ctakes.assertion.medfacts.cleartk.UncertaintyCleartkAnalysisEngine;
import org.apache.ctakes.chunker.ae.Chunker;
import org.apache.ctakes.chunker.ae.DefaultChunkCreator;
import org.apache.ctakes.chunker.ae.adjuster.ChunkAdjuster;
import org.apache.ctakes.consumers.CuisWriter;
import org.apache.ctakes.consumers.GranularJsonWriter;
import org.apache.ctakes.contexttokenizer.ae.ContextDependentTokenizerAnnotator;
import org.apache.ctakes.core.ae.OverlapAnnotator;
import org.apache.ctakes.core.ae.SentenceDetector;
import org.apache.ctakes.core.ae.SimpleSegmentAnnotator;
import org.apache.ctakes.core.ae.TokenizerAnnotatorPTB;
import org.apache.ctakes.core.resource.FileLocator;
import org.apache.ctakes.postagger.POSTagger;
import org.apache.ctakes.typesystem.type.structured.DocumentID;
import org.apache.ctakes.typesystem.type.syntax.BaseToken;
import org.apache.ctakes.typesystem.type.syntax.Chunk;
import org.apache.ctakes.typesystem.type.textspan.LookupWindowAnnotation;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.ctakes.utils.RushConfig;
import org.apache.ctakes.utils.Utils;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.TypePrioritiesFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.cleartk.util.ViewUriUtil;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

public class RushNiFiPipeline implements AutoCloseable {

    interface Options {

        @Option(longName = "input-dir")
        File getInputDirectory();

        @Option(longName = "output-dir")
        File getOutputDirectory();


        @Option(longName = "masterFolder")
        File getMasterFolder();

        @Option(longName = "tempMasterFolder")
        File getTempMasterFolder();
    }

    public static final String FAKE_DIR = "/tmp/random/";
    private static Gson gson = new Gson();

    private transient AnalysisEngine xmiAnnotationEngine;
    private transient AnalysisEngine cuisAnnotationConsumer;
    private transient AnalysisEngine granularAnnotationConsumer;
    private transient CollectionReader fileContentReader;
    private String lookupXmlPath;
    private String masterFolder;
    private boolean useDefaultForNegationAnnotations = true;

    public RushNiFiPipeline(RushConfig config, boolean useDefaultForNegationAnnotations) {
        try {
            this.lookupXmlPath = config.getLookupXml().getAbsolutePath();
            this.useDefaultForNegationAnnotations = useDefaultForNegationAnnotations;
            masterFolder = config.getMasterRoot().getAbsolutePath();
            xmiAnnotationEngine = getXMIWritingPreprocessorAggregateBuilder().createAggregate();
            cuisAnnotationConsumer = AnalysisEngineFactory.createEngine(CuisWriter.class);
            granularAnnotationConsumer = AnalysisEngineFactory.createEngine(GranularJsonWriter.class);
            fileContentReader = RushFilesCollectionReader.getCollectionReader(FAKE_DIR);


        } catch (Exception e) {
            // TODO Auto-generated catch block
            Throwables.propagate(e);
        }
    }

    String getCuis(final String xmi) throws Exception {
        CollectionReader xmlCollectionReader = Utils.getCollectionReader(xmi);
        return RushSimplePipeline.runPipeline(xmlCollectionReader, cuisAnnotationConsumer);
    }

    String getGranular(final String xmi) throws Exception {
        CollectionReader xmlCollectionReader = Utils.getCollectionReader(xmi);
        return RushSimplePipeline.runPipeline(xmlCollectionReader, granularAnnotationConsumer);
    }

    public CTakesResult getResult(String filePath, int partNo, String fileContent) throws Exception {
        String xml10pattern = "[^"
                + "\u0009\r\n"
                + "\u0020-\uD7FF"
                + "\uE000-\uFFFD"
                + "\ud800\udc00-\udbff\udfff"
                + "]";
        String legalFC = fileContent.replaceAll(xml10pattern, "");
        CAS rawFileCas = RushPipeline.initializeCas(fileContentReader, xmiAnnotationEngine);
        CTakesFilePart part = new CTakesFilePart(filePath, partNo, legalFC);
        fileContentReader.setConfigParameterValue("ctakesFilePart", part);
        CTakesResult result = RushPipeline.processCas(rawFileCas, fileContentReader, xmiAnnotationEngine);

        rawFileCas.reset();
        return result;
    }

    public void close() {
        try {
            RushPipeline.close(xmiAnnotationEngine);
            RushPipeline.close(cuisAnnotationConsumer);
        } catch (AnalysisEngineProcessException e) {
            // TODO Auto-generated catch block
            Throwables.propagate(e);
        }
    }

    public static void main(String[] args) throws Exception {
        Options options = CliFactory.parseArguments(Options.class, args);
        final File inputDirectory = options.getInputDirectory(); // text files to process
        final File outputDirectory = options.getOutputDirectory(); // directory to output xmi files
        final File masterFolder = options.getMasterFolder();
        final File tempMasterFolder = options.getTempMasterFolder();

        ensureCorrectSetup(masterFolder);

        try (RushConfig config = new RushConfig(masterFolder.getAbsolutePath(), tempMasterFolder.getAbsolutePath())) {
            config.initialize();
            try (RushNiFiPipeline pipeline = new RushNiFiPipeline(config, true)) {
                pipeline.execute(inputDirectory, outputDirectory);
            }
        }
    }

    void execute(File inputDirectory, File outputDirectory) throws Exception {
        for (File file : Objects.requireNonNull(inputDirectory.listFiles())) {
            String rawText = FileUtils.readFileToString(file);
            CTakesResult result = getResult(file.getAbsolutePath(), 1, rawText);

            String xmi = result.getOutput();
            String cuis = getCuis(xmi);
            String granular = getGranular(xmi);
            String overview = getOverview(rawText, xmi);

            FileUtils.write(new File(new File(outputDirectory, "xmis"), file.getName()), xmi);
            FileUtils.write(new File(new File(outputDirectory, "cuis"), file.getName()), cuis);
            FileUtils.write(new File(new File(outputDirectory, "granular"), file.getName()), granular);
            FileUtils.write(new File(new File(outputDirectory, "overview"), file.getName()), overview);
        }
    }

    private String getOverview(String rawText, String xmi) {
        Overview overview = new Overview();
        overview.rawText = rawText;
        overview.xmi = xmi;
        return gson.toJson(overview);
    }

    static class Overview {
        String fname = "";
        String loadID = "";
        String loadTimestamp = "";

        String rawText;
        String xmi;
    }

    private static void ensureCorrectSetup(File masterFolder) throws IOException {

        FileUtils.forceMkdir(new File(FAKE_DIR)); // required for current implementation...

        Path configLink = Paths.get("/tmp/ctakes-config");
        if (!Files.exists(configLink)) {
            Files.createSymbolicLink(configLink, masterFolder.toPath().toAbsolutePath());
        }
    }

    protected AggregateBuilder getXMIWritingPreprocessorAggregateBuilder() throws Exception {
        AggregateBuilder aggregateBuilder = new AggregateBuilder();
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(RushURIToDocumentTextAnnotator.class));

        // add document id annotation
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(DocumentIDAnnotator.class));

        // identify segments
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(SimpleSegmentAnnotator.class));

        // identify sentences
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(SentenceDetector.class,
                SentenceDetector.SD_MODEL_FILE_PARAM, "org/apache/ctakes/core/sentdetect/sd-med-model.zip"));
        // identify tokens
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(TokenizerAnnotatorPTB.class));
        // merge some tokens
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(ContextDependentTokenizerAnnotator.class));

        // identify part-of-speech tags
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(POSTagger.class,
                TypeSystemDescriptionFactory.createTypeSystemDescription(),
                TypePrioritiesFactory.createTypePriorities(Segment.class, Sentence.class, BaseToken.class),
                POSTagger.POS_MODEL_FILE_PARAM, "org/apache/ctakes/postagger/models/mayo-pos.zip"));

        // originally we had FileLocator.locateFile(
        // "org/apache/ctakes/chunker/models/chunker-model.zip" )
        // but this failed to locate the chunker model, so using the absolute path

        // String absolutePathToChunkerModel = System.getenv("CTAKES_HOME") +

        //String absolutePathToChunkerModel = "/tmp/ctakes-trunk/trunk/ctakes-chunker-res/src/main/resources/org/apache/ctakes/chunker/models/chunker-model.zip";
        String absolutePathToChunkerModel = masterFolder + "/org/apache/ctakes/chunker/models/chunker-model.zip";

        // "ctakes-chunker-res/src/main/resources/org/apache/ctakes/chunker/models/chunk-model.claims-1.5.zip";
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(Chunker.class,
                Chunker.CHUNKER_MODEL_FILE_PARAM, FileLocator.locateFile(absolutePathToChunkerModel),
                Chunker.CHUNKER_CREATOR_CLASS_PARAM, DefaultChunkCreator.class));

        // identify UMLS named entities

        // adjust NP in NP NP to span both
        aggregateBuilder.add(
                AnalysisEngineFactory.createEngineDescription(ChunkAdjuster.class, ChunkAdjuster.PARAM_CHUNK_PATTERN,
                        new String[]{"NP", "NP"}, ChunkAdjuster.PARAM_EXTEND_TO_INCLUDE_TOKEN, 1));
        // adjust NP in NP PP NP to span all three
        aggregateBuilder.add(
                AnalysisEngineFactory.createEngineDescription(ChunkAdjuster.class, ChunkAdjuster.PARAM_CHUNK_PATTERN,
                        new String[]{"NP", "PP", "NP"}, ChunkAdjuster.PARAM_EXTEND_TO_INCLUDE_TOKEN, 2));
        // add lookup windows for each NP
        aggregateBuilder
                .add(AnalysisEngineFactory.createEngineDescription(CopyNPChunksToLookupWindowAnnotations.class));
        // maximize lookup windows
        aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(OverlapAnnotator.class, "A_ObjectClass",
                LookupWindowAnnotation.class, "B_ObjectClass", LookupWindowAnnotation.class, "OverlapType", "A_ENV_B",
                "ActionType", "DELETE", "DeleteAction", new String[]{"selector=B"}));
        // add UMLS on top of lookup windows
        //aggregateBuilder.add(DefaultJCasTermAnnotator.createAnnotatorDescription(lookupXml.getAbsolutePath()));
        aggregateBuilder.add(RushDefaultJCasTermAnnotator.createAnnotatorDescription(lookupXmlPath));

        aggregateBuilder.add(MyLvgAnnotator.createAnnotatorDescription());

        // the following two AEs slow down the pipeline significantly when input file
        // are large
        if (this.useDefaultForNegationAnnotations) {
            aggregateBuilder.add(
                    PolarityCleartkAnalysisEngine.createAnnotatorDescription());
            aggregateBuilder.add(
                    UncertaintyCleartkAnalysisEngine.createAnnotatorDescription());
        } else {
            aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription("desc/NegationAnnotator"));
        }


        return aggregateBuilder;
    }

    /*
     * Add document id annotation
     */
    public static class DocumentIDAnnotator extends JCasAnnotator_ImplBase {

        @Override
        public void process(JCas jCas) throws AnalysisEngineProcessException {
            String documentID = new File(ViewUriUtil.getURI(jCas)).getPath();
            System.out.println("\nprocessing: " + documentID);
            DocumentID documentIDAnnotation = new DocumentID(jCas);
            documentIDAnnotation.setDocumentID(documentID);
            documentIDAnnotation.addToIndexes();
        }
    }


    public static class CopyNPChunksToLookupWindowAnnotations extends JCasAnnotator_ImplBase {

        @Override
        public void process(JCas jCas) {
            for (Chunk chunk : JCasUtil.select(jCas, Chunk.class)) {
                if (chunk.getChunkType().equals("NP")) {
                    new LookupWindowAnnotation(jCas, chunk.getBegin(), chunk.getEnd()).addToIndexes();
                }
            }
        }
    }

}
