package org.apache.ctakes.consumers;

import com.google.gson.Gson;
import org.apache.ctakes.typesystem.type.refsem.OntologyConcept;
import org.apache.ctakes.typesystem.type.refsem.UmlsConcept;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;
import org.apache.uima.cas.FeatureStructure;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;

import java.util.ArrayList;
import java.util.List;

public class GranularJsonWriter extends JCasAnnotator_ImplBase {

    private static Gson gson = new Gson();
    private String result = "";

    public String getResult() {
        return result;
    }

    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {

        try {
            jCas.getView("_InitialView");
        } catch (CASException e) {
            throw new AnalysisEngineProcessException(e);
        }

        List<GranularInfo> granularInfoList = new ArrayList<>();

        for (EventMention eventMention : JCasUtil.select(jCas, EventMention.class)) {

            int polarity = eventMention.getPolarity();
            int uncertainty = eventMention.getUncertainty();
            float confidence = eventMention.getConfidence();
            int startPosition = eventMention.getBegin();
            int endPosition = eventMention.getEnd();

            String coveredText = eventMention.getCoveredText();

            FeatureStructure[] featureStructures = eventMention.getOntologyConceptArr().toArray();

            for (FeatureStructure featureStructure : featureStructures) {
                OntologyConcept ontologyConcept = (OntologyConcept) featureStructure;

                GranularInfo gi = new GranularInfo();
                granularInfoList.add(gi);

                gi.polarity = polarity;
                gi.confidence = confidence;
                gi.uncertainty = uncertainty;
                gi.startPosition = startPosition;
                gi.endPosition = endPosition;
                gi.coveredText = coveredText;

                gi.codingScheme = ontologyConcept.getCodingScheme();
                gi.code = ontologyConcept.getCode();

                if (ontologyConcept instanceof UmlsConcept) {
                    UmlsConcept umlsConcept = (UmlsConcept) ontologyConcept;
                    gi.cui = umlsConcept.getCui();
                    gi.tui = umlsConcept.getTui();
                    gi.preferredText = umlsConcept.getPreferredText();
                }
            }
        }

        String outputString = gson.toJson(granularInfoList);

        try {
            jCas.createView("RESULT_VIEW").setDocumentText(outputString);
        } catch (CASException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        this.result = outputString;
    }

    static class GranularInfo {
        String fname = "";
        String loadID = "";
        String loadTimestamp = "";

        int polarity;
        float confidence;
        int uncertainty;
        int startPosition;
        int endPosition;
        String coveredText;

        String codingScheme;
        String code;
        String cui;
        String tui;
        String preferredText;
    }
}