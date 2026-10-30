package org.apache.commons.csv;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.StringReader;

import org.junit.jupiter.api.Test;

public class CSVFormatTest {

    @Test
    public void testPrintWithQuotesMinimalPlain() throws IOException {
        { final StringBuilder out = new StringBuilder(); CSVFormat.DEFAULT.print("abc", out, true); assertEquals("abc", out.toString()); }
    }

    @Test
    public void testPrintWithQuotesMinimalDelimiter() throws IOException {
        { final StringBuilder out = new StringBuilder(); CSVFormat.DEFAULT.print("a,b", out, true); assertEquals("\"a,b\"", out.toString()); }
    }

    @Test
    public void testPrintWithQuotesEmbeddedQuote() throws IOException {
        { final StringBuilder out = new StringBuilder(); CSVFormat.DEFAULT.print("a\"b", out, true); assertEquals("\"a\"\"b\"", out.toString()); }
    }

    @Test
    public void testPrintWithQuotesAll() throws IOException {
        final CSVFormat format = CSVFormat.DEFAULT.builder().setQuoteMode(QuoteMode.ALL).build();
        { final StringBuilder out = new StringBuilder(); format.print("x", out, true); assertEquals("\"x\"", out.toString()); }
    }

    @Test
    public void testPrintWithQuotesNonNumeric() throws IOException {
        final CSVFormat format = CSVFormat.DEFAULT.builder().setQuoteMode(QuoteMode.NON_NUMERIC).build();
        { final StringBuilder out = new StringBuilder(); format.print(Integer.valueOf(1), out, true); assertEquals("1", out.toString()); }
        { final StringBuilder out = new StringBuilder(); format.print("y", out, true); assertEquals("\"y\"", out.toString()); }
    }

    @Test
    public void testPrintWithQuotesEmptyFirstValue() throws IOException {
        { final StringBuilder out = new StringBuilder(); CSVFormat.DEFAULT.print("", out, true); assertEquals("\"\"", out.toString()); }
    }

    @Test
    public void testPrintWithQuotesReader() throws IOException {
        final StringBuilder out = new StringBuilder();
        CSVFormat.DEFAULT.print(new StringReader("q\"r"), out, true);
        assertEquals("\"q\"\"r\"", out.toString());
    }
}
