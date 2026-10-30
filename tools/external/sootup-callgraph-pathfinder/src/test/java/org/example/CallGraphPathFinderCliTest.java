package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CallGraphPathFinderCliTest {

    @Test
    void parsesRequiredPositionalArgumentsOnly() {
        String[] args = {
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(CallGraphPathFinder.RootMode.PUBLIC, cfg.mode);
        assertTrue(cfg.limitPackageRoots);
        assertNull(cfg.depDir);
        assertNull(cfg.outputJson);
        assertEquals(0, cfg.maxPaths);
        assertEquals(0L, cfg.maxSearchStates);
        assertEquals("cp-main.jar", cfg.classPath);
        assertEquals("com.example.Target", cfg.targetClass);
        assertEquals("targetMethod", cfg.targetMethodName);
        assertEquals(3, cfg.maxDepth);
        assertEquals("call_paths.txt", cfg.outputFile);
    }

    @Test
    void parsesAllOptionalFlagsAndPositionals() {
        String[] args = {
                "--roots=ALL",
                "--limitPackageRoots=false",
                "--depDir=/tmp/deps",
                "--outputJson=/tmp/out.json",
                "--maxPaths=150",
                "--maxSearchStates=100000",
                "cp-main.jar" + java.io.File.pathSeparator + "cp-lib.jar",
                "com.example.Target",
                "targetMethod",
                "7",
                "custom_output.txt"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(CallGraphPathFinder.RootMode.ALL, cfg.mode);
        assertFalse(cfg.limitPackageRoots);
        assertEquals("/tmp/deps", cfg.depDir);
        assertEquals("/tmp/out.json", cfg.outputJson);
        assertEquals(150, cfg.maxPaths);
        assertEquals(100000L, cfg.maxSearchStates);
        assertEquals(7, cfg.maxDepth);
        assertEquals("custom_output.txt", cfg.outputFile);
    }

    @Test
    void defaultsInvalidLimitValuesToUnlimited() {
        String[] args = {
                "--maxPaths=oops",
                "--maxSearchStates=nope",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(0, cfg.maxPaths);
        assertEquals(0L, cfg.maxSearchStates);
    }

    @Test
    void normalizesNegativeLimitValuesToUnlimited() {
        String[] args = {
                "--maxPaths=-5",
                "--maxSearchStates=-9",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(0, cfg.maxPaths);
        assertEquals(0L, cfg.maxSearchStates);
    }

    @Test
    void returnsNullWhenMissingRequiredArguments() {
        String[] args = {
                "--roots=PUBLIC",
                "only-classpath"
        };
        assertNull(CallGraphPathFinder.parseCliArgs(args));
    }

    @Test
    void interpretsFourthPositionalAsOutputFileWhenNotNumeric() {
        String[] args = {
                "cp-main.jar",
                "com.example.Target",
                "targetMethod",
                "my_output.txt"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(3, cfg.maxDepth);
        assertEquals("my_output.txt", cfg.outputFile);
    }

    @Test
    void interpretsThreadsParameterCorrectly() {
        String[] args = {
                "--threads=4",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(4, cfg.threads);
    }

    @Test
    void defaultsThreadsToOne() {
        String[] args = {
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(1, cfg.threads);
    }

    @Test
    void normalizesInvalidThreadsValuesToOne() {
        String[] args = {
                "--threads=invalid",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(1, cfg.threads);
    }

    @Test
    void normalizesZeroOrNegativeThreadsToOne() {
        String[] args1 = {
                "--threads=0",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg1 = CallGraphPathFinder.parseCliArgs(args1);
        assertNotNull(cfg1);
        assertEquals(1, cfg1.threads);

        String[] args2 = {
                "--threads=-5",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod"
        };

        CallGraphPathFinder.CliConfig cfg2 = CallGraphPathFinder.parseCliArgs(args2);
        assertNotNull(cfg2);
        assertEquals(1, cfg2.threads);
    }

    @Test
    void interpretsMultipleThreadsAndOtherOptions() {
        String[] args = {
                "--maxPaths=50",
                "--threads=8",
                "--maxSearchStates=50000",
                "cp-main.jar",
                "com.example.Target",
                "targetMethod",
                "5",
                "out.txt"
        };

        CallGraphPathFinder.CliConfig cfg = CallGraphPathFinder.parseCliArgs(args);
        assertNotNull(cfg);
        assertEquals(50, cfg.maxPaths);
        assertEquals(8, cfg.threads);
        assertEquals(50000L, cfg.maxSearchStates);
        assertEquals(5, cfg.maxDepth);
        assertEquals("out.txt", cfg.outputFile);
    }
}
