package org.apache.commons.csv;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.StringReader;

import org.junit.jupiter.api.Test;

public class LexerTest {

    @Test
    public void testNextTokenSimpleTokens() throws IOException {
        try (Lexer lexer = new Lexer(CSVFormat.DEFAULT, new ExtendedBufferedReader(new StringReader("a,b\nc")))) {
            Token t = lexer.nextToken(new Token());
            assertEquals(Token.Type.TOKEN, t.type);
            assertEquals("a", t.content.toString());
            t = lexer.nextToken(new Token());
            assertEquals(Token.Type.EORECORD, t.type);
            assertEquals("b", t.content.toString());
            t = lexer.nextToken(new Token());
            assertEquals(Token.Type.EOF, t.type);
            assertEquals("c", t.content.toString());
        }
    }

    @Test
    public void testNextTokenQuotedToken() throws IOException {
        try (Lexer lexer = new Lexer(CSVFormat.DEFAULT, new ExtendedBufferedReader(new StringReader("\"x,y\",z")))) {
            final Token t = lexer.nextToken(new Token());
            assertEquals(Token.Type.TOKEN, t.type);
            assertEquals("x,y", t.content.toString());
        }
    }

    @Test
    public void testNextTokenComment() throws IOException {
        final CSVFormat format = CSVFormat.DEFAULT.builder().setCommentMarker('#').build();
        try (Lexer lexer = new Lexer(format, new ExtendedBufferedReader(new StringReader("# note\na")))) {
            final Token t = lexer.nextToken(new Token());
            assertEquals(Token.Type.COMMENT, t.type);
            assertEquals("note", t.content.toString());
        }
    }

    @Test
    public void testNextTokenIgnoreEmptyLines() throws IOException {
        try (Lexer lexer = new Lexer(CSVFormat.DEFAULT, new ExtendedBufferedReader(new StringReader("\n\nfoo")))) {
            final Token t = lexer.nextToken(new Token());
            assertEquals("foo", t.content.toString());
        }
    }
}
