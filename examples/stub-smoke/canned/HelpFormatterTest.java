package org.apache.commons.cli;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class HelpFormatterTest {

    @Test
    public void testRenderOptionsShortAndLong() {
        final Options options = new Options();
        options.addOption("a", "all", false, "show all entries");
        options.addOption(Option.builder().longOpt("size").hasArg().argName("SIZE").desc("block size").build());
        final HelpFormatter formatter = new HelpFormatter();
        final String rendered = formatter.renderOptions(new StringBuffer(), 80, options, 1, 3).toString();
        assertTrue(rendered.contains("-a,--all"));
        assertTrue(rendered.contains("--size <SIZE>"));
        assertTrue(rendered.contains("block size"));
    }

    @Test
    public void testRenderOptionsWrapsLongDescription() {
        final Options options = new Options();
        options.addOption("x", "extra", true, "a very long description that needs to be wrapped across multiple lines of output");
        final HelpFormatter formatter = new HelpFormatter();
        formatter.setOptionComparator(null);
        final String rendered = formatter.renderOptions(new StringBuffer(), 40, options, 0, 2).toString();
        assertTrue(rendered.split(System.lineSeparator()).length > 1);
    }
}
