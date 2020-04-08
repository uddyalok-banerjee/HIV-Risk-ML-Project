package org.apache.ctakes.pipelines;

import com.google.common.base.Throwables;
import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;
import org.apache.commons.io.FileUtils;
import org.apache.ctakes.assertion.medfacts.cleartk.PolarityCleartkAnalysisEngine;
import org.apache.ctakes.assertion.medfacts.cleartk.UncertaintyCleartkAnalysisEngine;
import org.apache.ctakes.chunker.ae.Chunker;
import org.apache.ctakes.chunker.ae.DefaultChunkCreator;
import org.apache.ctakes.chunker.ae.adjuster.ChunkAdjuster;
import org.apache.ctakes.clinicalpipeline.ClinicalPipelineFactory;
import org.apache.ctakes.consumers.CuisWriter;
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
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.TypePrioritiesFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.XMLSerializer;
import org.cleartk.util.ViewUriUtil;
import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class RushNiFiPipeline {
	public static final String MASTER_FILE_NAME="db.xml";
	public static final String TEMPLATE_STRING="$CTAKES_ROOT$";

	static interface Options {

		@Option(longName = "input-dir")
		public File getInputDirectory();

		@Option(longName = "output-dir")
		public File getOutputDirectory();




		@Option(longName= "masterFolder")
		public File getMasterFolder();

		@Option(longName= "tempMasterFolder")
		public File getTempMasterFolder();
	}

	public static File inputDirectory; // text files to process
	public static File outputDirectory; // directory to output xmi files
	//public static File lookupXml;
	//public static File masterFolder;
	public  File tempMasterFolder;


	public static final String FAKE_DIR = "/tmp/random/";

	private transient AnalysisEngine xmiAnnotationEngine;
	private transient AnalysisEngine cuisAnnotationConsumer;
	private transient CollectionReader fileContentReader;
	private transient CAS rawFileCas;
	private  String lookupXmlPath;
	private String masterFolder;
	private boolean useDefaultForNegationAnnotations = true;

	public RushNiFiPipeline(RushConfig config, boolean useDefaultForNegationAnnotations) {
		try {
			this.lookupXmlPath = config.getLookupXml().getAbsolutePath();
			this.useDefaultForNegationAnnotations = useDefaultForNegationAnnotations;
			masterFolder = config.getMasterRoot().getAbsolutePath();
			xmiAnnotationEngine = getXMIWritingPreprocessorAggregateBuilder().createAggregate();
			cuisAnnotationConsumer = AnalysisEngineFactory.createEngine(CuisWriter.class);
			fileContentReader = RushFilesCollectionReader.getCollectionReader(FAKE_DIR);
			

			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			Throwables.propagate(e);
		}
	}
	
	public CTakesResult getResult(String filePath, int partNo, String fileContent) throws Exception {
		String xml10pattern = "[^"
                + "\u0009\r\n"
                + "\u0020-\uD7FF"
                + "\uE000-\uFFFD"
                + "\ud800\udc00-\udbff\udfff"
                + "]";
		String legalFC = fileContent.replaceAll(xml10pattern, "");
		rawFileCas = RushPipeline.initializeCas(fileContentReader, xmiAnnotationEngine);
		CTakesFilePart part = new CTakesFilePart(filePath, partNo, legalFC);
		fileContentReader.setConfigParameterValue("ctakesFilePart", part);
		CTakesResult result = RushPipeline.processCas(rawFileCas, fileContentReader, xmiAnnotationEngine);

		CollectionReader xmlCollectionReader = Utils.getCollectionReader(result.getOutput());

		String cuis = RushSimplePipeline.runPipeline(xmlCollectionReader, cuisAnnotationConsumer);
		result.setCuis(cuis);

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
		inputDirectory = options.getInputDirectory();
		outputDirectory = options.getOutputDirectory();
		//lookupXml = options.getLookupXml();
		File masterFolder = options.getMasterFolder();
		File tempMasterFolder = options.getTempMasterFolder();

		RushConfig config = new RushConfig(masterFolder.getAbsolutePath(),
				tempMasterFolder.getAbsolutePath()); 
		config.initialize();
		RushNiFiPipeline pipeline = new RushNiFiPipeline(config,true);

		for (File file : inputDirectory.listFiles()) {
			String t = FileUtils.readFileToString(file);
			CTakesResult result = pipeline.getResult(file.getAbsolutePath(), 1, t);
			FileUtils.write(new File(new File(outputDirectory,"xmis"),file.getName()), result.getOutput());
			FileUtils.write(new File(new File(outputDirectory,"cuis"),file.getName()), result.getCuis());
		}
		System.out.println("Closing Pipeline");
		pipeline.close();
		config.close();
		//FileUtils.deleteDirectory(newConfigFolder);
		System.out.println("Closed Pipeline, Now Exiting");
	}

	

	protected static AggregateBuilder getFastPipeline() throws Exception {

		AggregateBuilder aggregateBuilder = new AggregateBuilder();

		aggregateBuilder.add(ClinicalPipelineFactory.getFastPipeline());

		return aggregateBuilder;
	}

	protected  AggregateBuilder getXMIWritingPreprocessorAggregateBuilder() throws Exception {
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
						new String[] { "NP", "NP" }, ChunkAdjuster.PARAM_EXTEND_TO_INCLUDE_TOKEN, 1));
		// adjust NP in NP PP NP to span all three
		aggregateBuilder.add(
				AnalysisEngineFactory.createEngineDescription(ChunkAdjuster.class, ChunkAdjuster.PARAM_CHUNK_PATTERN,
						new String[] { "NP", "PP", "NP" }, ChunkAdjuster.PARAM_EXTEND_TO_INCLUDE_TOKEN, 2));
		// add lookup windows for each NP
		aggregateBuilder
				.add(AnalysisEngineFactory.createEngineDescription(CopyNPChunksToLookupWindowAnnotations.class));
		// maximize lookup windows
		aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(OverlapAnnotator.class, "A_ObjectClass",
				LookupWindowAnnotation.class, "B_ObjectClass", LookupWindowAnnotation.class, "OverlapType", "A_ENV_B",
				"ActionType", "DELETE", "DeleteAction", new String[] { "selector=B" }));
		// add UMLS on top of lookup windows
		//aggregateBuilder.add(DefaultJCasTermAnnotator.createAnnotatorDescription(lookupXml.getAbsolutePath()));
		aggregateBuilder.add(RushDefaultJCasTermAnnotator.createAnnotatorDescription(lookupXmlPath));
		
		aggregateBuilder.add(MyLvgAnnotator.createAnnotatorDescription());

		// the following two AEs slow down the pipeline significantly when input file
		// are large
		if(this.useDefaultForNegationAnnotations) {
			aggregateBuilder.add(
					 PolarityCleartkAnalysisEngine.createAnnotatorDescription() );
					 aggregateBuilder.add(
					 UncertaintyCleartkAnalysisEngine.createAnnotatorDescription() );
		}else {
			aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription("desc/NegationAnnotator"));
		}


		

		// write out the CAS after all the above annotations //Write to Spark Output
		// context directly from here
		/*
		 * AnalysisEngineDescription writer =
		 * AnalysisEngineFactory.createEngineDescription( HiveWriter.class,
		 * HiveWriter.PARAM_XMI_DIRECTORY, outputDirectory);
		 * 
		 * aggregateBuilder.add(writer); ß
		 */
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

	/*
	 * The following class overrides a ClearTK utility annotator class for reading a
	 * text file into a JCas. The code is copy/pasted so that one tiny modification
	 * can be made for this corpus -- replace a single odd character (0xc) with a
	 * space since it trips up xml output.
	 */
//  public static class UriToDocumentTextAnnotatorCtakes extends UriToDocumentTextAnnotator {
//
//    @Override
//    public void process(JCas jCas) throws AnalysisEngineProcessException {
//      URI uri = ViewUriUtil.getURI(jCas);
//      String content;
//
//      try {
//        content = CharStreams.toString(new InputStreamReader(uri.toURL().openStream()));
//        content = content.replace((char) 0xc, ' ');
//        jCas.setSofaDataString(content, "text/plain");
//      } catch (MalformedURLException e) {
//        throw new AnalysisEngineProcessException(e);
//      } catch (IOException e) {
//        throw new AnalysisEngineProcessException(e);
//      }
//    }  
//  }

	public static class CopyNPChunksToLookupWindowAnnotations extends JCasAnnotator_ImplBase {

		@Override
		public void process(JCas jCas) throws AnalysisEngineProcessException {
			for (Chunk chunk : JCasUtil.select(jCas, Chunk.class)) {
				if (chunk.getChunkType().equals("NP")) {
					new LookupWindowAnnotation(jCas, chunk.getBegin(), chunk.getEnd()).addToIndexes();
				}
			}
		}
	}

	public static class XMIWriter extends JCasAnnotator_ImplBase {

		public static final String PARAM_XMI_DIRECTORY = "XMIDirectory";

		@ConfigurationParameter(name = PARAM_XMI_DIRECTORY, mandatory = true)
		private File xmiDirectory;

		@Override
		public void initialize(UimaContext context) throws ResourceInitializationException {
			super.initialize(context);
			if (!this.xmiDirectory.exists()) {
				this.xmiDirectory.mkdirs();
			}
		}

		@Override
		public void process(JCas jCas) throws AnalysisEngineProcessException {
			File xmiFile = getXMIFile(this.xmiDirectory, jCas);
			try {
				FileOutputStream outputStream = new FileOutputStream(xmiFile);
				try {
					XmiCasSerializer serializer = new XmiCasSerializer(jCas.getTypeSystem());
					ContentHandler handler = new XMLSerializer(outputStream, false).getContentHandler();
					serializer.serialize(jCas.getCas(), handler);
				} finally {
					outputStream.close();
				}
			} catch (SAXException e) {
				throw new AnalysisEngineProcessException(e);
			} catch (IOException e) {
				throw new AnalysisEngineProcessException(e);
			}
		}

		static File getXMIFile(File xmiDirectory, JCas jCas) throws AnalysisEngineProcessException {
			return getXMIFile(xmiDirectory, new File(ViewUriUtil.getURI(jCas).getPath()));
		}

		static File getXMIFile(File xmiDirectory, File textFile) {
			return new File(xmiDirectory, textFile.getName() + ".xmi");
		}
	}

}
