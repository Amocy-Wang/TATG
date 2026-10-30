package org.example;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class PathfindingResultSerializerTest {

    @Test
    void jsonStringContainsAllRequiredFields() {
        PathfindingResultSerializer.SerializationRequest req = new PathfindingResultSerializer.SerializationRequest();
        req.targetClass = "com.example.Target";
        req.targetMethod = "targetMethod";
        req.rootMode = "PUBLIC";
        req.maxDepth = 5;
        req.limitPackageRoots = true;
        req.maxPaths = 100;
        req.maxSearchStates = 50000;
        req.pathCount = 3;
        req.indirectPathCount = 1;
        req.initializationLineCount = 2;
        req.exploredStates = 5423;
        req.emittedPathCandidates = 18;
        req.truncatedByPathLimit = false;
        req.truncatedByStateLimit = false;
        req.textOutputFile = "call_paths.txt";
        req.pathSteps = new ArrayList<>();
        req.pathSteps.add(List.of("step1", "step2"));
        req.initializationLines = new HashSet<>(Set.of("init1", "init2"));

        String json = PathfindingResultSerializer.buildJsonString(req);

        // Verify top-level structure
        assertJsonContains(json, "\"target\"");
        assertJsonContains(json, "\"config\"");
        assertJsonContains(json, "\"stats\"");
        assertJsonContains(json, "\"artifacts\"");
        assertJsonContains(json, "\"paths\"");
        assertJsonContains(json, "\"initializationLines\"");

        // Verify target section
        assertJsonContains(json, "\"class\": \"com.example.Target\"");
        assertJsonContains(json, "\"method\": \"targetMethod\"");

        // Verify config section
        assertJsonContains(json, "\"rootMode\": \"PUBLIC\"");
        assertJsonContains(json, "\"maxDepth\": 5");
        assertJsonContains(json, "\"maxPaths\": 100");
        assertJsonContains(json, "\"maxSearchStates\": 50000");

        // Verify stats section (key for truncation tracking)
        assertJsonContains(json, "\"exploredStates\": 5423");
        assertJsonContains(json, "\"emittedPathCandidates\": 18");
        assertJsonContains(json, "\"truncatedByPathLimit\": false");
        assertJsonContains(json, "\"truncatedByStateLimit\": false");

        // Verify artifacts and counts
        assertJsonContains(json, "\"pathCount\": 3");
        assertJsonContains(json, "\"indirectPathCount\": 1");
        assertJsonContains(json, "\"initializationLineCount\": 2");
    }

    @Test
    void jsonStringCorrectlySerializesTruncationFlags() {
        PathfindingResultSerializer.SerializationRequest req = createMinimalRequest();
        req.truncatedByPathLimit = true;
        req.truncatedByStateLimit = false;
        req.exploredStates = 100000;
        req.emittedPathCandidates = 9999;

        String json = PathfindingResultSerializer.buildJsonString(req);
        assertJsonContains(json, "\"truncatedByPathLimit\": true");
        assertJsonContains(json, "\"truncatedByStateLimit\": false");
        assertJsonContains(json, "\"exploredStates\": 100000");
        assertJsonContains(json, "\"emittedPathCandidates\": 9999");
    }

    @Test
    void jsonStringHandlesSpecialCharactersInEscapeStrings() {
        PathfindingResultSerializer.SerializationRequest req = createMinimalRequest();
        req.targetClass = "com.example.\"Quoted\"";
        req.targetMethod = "method\nWith\tSpecial";
        req.textOutputFile = "path\\to\\file.txt";

        String json = PathfindingResultSerializer.buildJsonString(req);
        // Should escape without breaking JSON structure
        assertTrue(json.contains("com.example.\\\"Quoted\\\""));
        assertTrue(json.contains("method\\nWith\\tSpecial"));
        assertTrue(json.contains("path\\\\to\\\\file.txt"));
        // JSON should still be valid structure
        assertTrue(json.contains("\"target\"") && json.contains("\"class\""));
    }

    @Test
    void jsonStringPreservesPathStructure() {
        PathfindingResultSerializer.SerializationRequest req = createMinimalRequest();
        req.pathSteps.add(List.of("A -> B", "B -> C", "C -> D"));
        req.pathSteps.add(List.of("X -> Y"));

        String json = PathfindingResultSerializer.buildJsonString(req);
        // Verify paths array structure
        assertJsonContains(json, "\"paths\": [");
        assertJsonContains(json, "\"length\": 3");
        assertJsonContains(json, "\"length\": 1");
        assertJsonContains(json, "\"indirect\": true");   // length > 1
        assertJsonContains(json, "\"indirect\": false");  // length == 1
        assertJsonContains(json, "\"A -> B\"");
        assertJsonContains(json, "\"X -> Y\"");
    }

    @Test
    void jsonStringHandlesEmptyPathsAndInitLines() {
        PathfindingResultSerializer.SerializationRequest req = createMinimalRequest();
        req.pathSteps = new ArrayList<>();
        req.initializationLines = new HashSet<>();

        String json = PathfindingResultSerializer.buildJsonString(req);
        assertJsonContains(json, "\"paths\": [");
        assertJsonContains(json, "\"initializationLines\": []");
    }

    @Test
    void jsonEscapeHandlesAllSpecialChars() {
        String escaped = PathfindingResultSerializer.jsonEscape("a\\b\"c\nd\re\tf");
        assertEquals("a\\\\b\\\"c\\nd\\re\\tf", escaped);
    }

    // Helper methods
    private PathfindingResultSerializer.SerializationRequest createMinimalRequest() {
        PathfindingResultSerializer.SerializationRequest req = new PathfindingResultSerializer.SerializationRequest();
        req.targetClass = "Test";
        req.targetMethod = "test";
        req.rootMode = "PUBLIC";
        req.maxDepth = 3;
        req.limitPackageRoots = true;
        req.maxPaths = 0;
        req.maxSearchStates = 0;
        req.pathCount = 0;
        req.indirectPathCount = 0;
        req.initializationLineCount = 0;
        req.exploredStates = 0;
        req.emittedPathCandidates = 0;
        req.truncatedByPathLimit = false;
        req.truncatedByStateLimit = false;
        req.textOutputFile = "out.txt";
        req.pathSteps = new ArrayList<>();
        req.initializationLines = new HashSet<>();
        return req;
    }

    private void assertJsonContains(String json, String expected) {
        assertTrue(json.contains(expected), "JSON does not contain: " + expected + "\nFull JSON:\n" + json);
    }
}
