package org.example;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Handles serialization of callgraph pathfinding results to JSON format.
 * Encapsulates JSON field mapping and formatting logic for maintainability.
 */
public class PathfindingResultSerializer {

    public static class SerializationRequest {
        public String targetClass;
        public String targetMethod;
        public String rootMode;
        public int maxDepth;
        public boolean limitPackageRoots;
        public int maxPaths;
        public long maxSearchStates;
        public int pathCount;
        public int indirectPathCount;
        public int initializationLineCount;
        public long exploredStates;
        public long emittedPathCandidates;
        public boolean truncatedByPathLimit;
        public boolean truncatedByStateLimit;
        public String textOutputFile;
        public List<List<String>> pathSteps;  // List of paths, each path is a list of steps
        public Set<String> initializationLines;
    }

    /**
     * Serializes the given request to JSON and writes to the specified output file.
     */
    public static void serializeToJson(SerializationRequest req, String jsonOutputFile) throws IOException {
        String json = buildJsonString(req);
        Path out = Paths.get(jsonOutputFile);
        Path parent = out.getParent();
        if (parent != null) Files.createDirectories(parent);
        Files.writeString(out, json, StandardCharsets.UTF_8);
    }

    /**
     * Builds the JSON string representation of the pathfinding result.
     */
    static String buildJsonString(SerializationRequest req) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"target\": {\n");
        sb.append("    \"class\": \"").append(jsonEscape(req.targetClass)).append("\",\n");
        sb.append("    \"method\": \"").append(jsonEscape(req.targetMethod)).append("\"\n");
        sb.append("  },\n");
        sb.append("  \"config\": {\n");
        sb.append("    \"rootMode\": \"").append(req.rootMode).append("\",\n");
        sb.append("    \"maxDepth\": ").append(req.maxDepth).append(",\n");
        sb.append("    \"limitPackageRoots\": ").append(req.limitPackageRoots).append(",\n");
        sb.append("    \"maxPaths\": ").append(req.maxPaths).append(",\n");
        sb.append("    \"maxSearchStates\": ").append(req.maxSearchStates).append("\n");
        sb.append("  },\n");
        sb.append("  \"stats\": {\n");
        sb.append("    \"pathCount\": ").append(req.pathCount).append(",\n");
        sb.append("    \"indirectPathCount\": ").append(req.indirectPathCount).append(",\n");
        sb.append("    \"initializationLineCount\": ").append(req.initializationLineCount).append(",\n");
        sb.append("    \"exploredStates\": ").append(req.exploredStates).append(",\n");
        sb.append("    \"emittedPathCandidates\": ").append(req.emittedPathCandidates).append(",\n");
        sb.append("    \"truncatedByPathLimit\": ").append(req.truncatedByPathLimit).append(",\n");
        sb.append("    \"truncatedByStateLimit\": ").append(req.truncatedByStateLimit).append("\n");
        sb.append("  },\n");
        sb.append("  \"artifacts\": {\n");
        sb.append("    \"textOutput\": \"").append(jsonEscape(req.textOutputFile)).append("\"\n");
        sb.append("  },\n");
        sb.append("  \"paths\": [\n");
        for (int i = 0; i < req.pathSteps.size(); i++) {
            List<String> steps = req.pathSteps.get(i);
            sb.append("    {\n");
            sb.append("      \"length\": ").append(steps.size()).append(",\n");
            sb.append("      \"indirect\": ").append(steps.size() > 1).append(",\n");
            sb.append("      \"steps\": [");
            for (int j = 0; j < steps.size(); j++) {
                if (j > 0) sb.append(", ");
                sb.append("\"").append(jsonEscape(steps.get(j))).append("\"");
            }
            sb.append("]\n");
            sb.append("    }");
            if (i < req.pathSteps.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ],\n");
        sb.append("  \"initializationLines\": [");
        for (String line : req.initializationLines) {
            sb.append("\"").append(jsonEscape(line)).append("\", ");
        }
        if (!req.initializationLines.isEmpty()) {
            sb.setLength(sb.length() - 2);  // Remove trailing ", "
        }
        sb.append("]\n");
        sb.append("}\n");
        return sb.toString();
    }

    /**
     * Escapes special characters for JSON string values.
     */
    static String jsonEscape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
