package org.apache.ctakes.utils;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.curator.shaded.com.google.common.base.Throwables;
import org.apache.tools.ant.util.StringUtils;

public class RushConfig implements AutoCloseable {
    public static final String MASTER_FILE_NAME = "db.xml";
    public static final String TEMPLATE_STRING = "$CTAKES_ROOT$";
    public static final String MASTER_TEMPLATIZED_LOOKUP_XML = "sno_rx_16ab-test.xml";
    private File masterRoot;
    private File tmpConfigRoot;
    private File newConfigFolder;

    public RushConfig(String masterRoot, String tmpConfigRoot) {
        this.masterRoot = new File(masterRoot);
        this.tmpConfigRoot = new File(tmpConfigRoot);
    }

    public File getMasterRoot() {
        return masterRoot;
    }

    public File getTmpConfigRoot() {
        return tmpConfigRoot;
    }

    public File getNewConfigFolder() {
        return newConfigFolder;
    }

    public File getLookupXml() {
        return new File(this.newConfigFolder, MASTER_FILE_NAME);
    }

    public File initialize() {
        try {

            String randomPrefix = Long.toString(Math.abs((new Random()).nextLong()));
            System.err.println(randomPrefix);
            if (!this.tmpConfigRoot.exists()) {
                FileUtils.forceMkdir(this.tmpConfigRoot);
            }
            newConfigFolder = new File(this.tmpConfigRoot, randomPrefix);
            if (newConfigFolder.exists()) {
                FileUtils.deleteDirectory(newConfigFolder);
            }
            FileUtils.forceMkdir(newConfigFolder);
            FileUtils.copyDirectory(this.masterRoot, newConfigFolder);


            String fContents = FileUtils.readFileToString(new File(this.masterRoot, MASTER_TEMPLATIZED_LOOKUP_XML));
            File newLookupXml = new File(newConfigFolder, MASTER_FILE_NAME);
            FileUtils.write(newLookupXml, StringUtils.replace(fContents, TEMPLATE_STRING, newConfigFolder.getAbsolutePath()));
            return newLookupXml;
        } catch (Exception ex) {
            Throwables.propagate(ex);
        }
        return null;
    }

    public void close() {
        try {
            FileUtils.deleteDirectory(newConfigFolder);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            Throwables.propagate(e);
        }
    }

    public static void main(String[] args) {
        RushConfig r = new RushConfig("/tmp/ctakes-config/", "/tmp/ctakes-config2/");
        r.initialize();
        r.close();
    }
}
