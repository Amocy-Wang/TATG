package org.example;

import sootup.callgraph.CallGraph;
import sootup.callgraph.CallGraphAlgorithm;
import sootup.callgraph.ClassHierarchyAnalysisAlgorithm;
import sootup.core.inputlocation.AnalysisInputLocation;
import sootup.core.jimple.common.expr.AbstractInvokeExpr;
import sootup.core.model.SootClass;
import sootup.core.model.SootMethod;
import sootup.core.model.SourceType;
import sootup.core.signatures.MethodSignature;
import sootup.java.bytecode.frontend.inputlocation.DefaultRuntimeAnalysisInputLocation;
import sootup.java.bytecode.frontend.inputlocation.JavaClassPathAnalysisInputLocation;
import sootup.java.core.views.JavaView;

import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Consumer;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.Collectors;

public class CallGraphPathFinder {
    enum RootMode { ALL, PUBLIC, MAINS }
    private static final String USAGE =
            "Usage: [--roots=ALL|PUBLIC|MAINS] [--limitPackageRoots=true|false(default:true)] [--depDir=<path>] "
                    + "[--outputJson=<path>] [--maxPaths=<n>] [--maxSearchStates=<n>] [--threads=<n>(default:1)] "
                    + "<classpath> <targetClassFQN> <targetMethodName> [maxDepth=3] [outputFile]";

    static final class CliConfig {
        RootMode mode = RootMode.PUBLIC;
        boolean limitPackageRoots = true;
        String depDir = null;
        String outputJson = null;
        int maxPaths = 0;
        long maxSearchStates = 0;
        int threads = 1;
        String classPath;
        String targetClass;
        String targetMethodName;
        int maxDepth = 3;
        String outputFile = "call_paths.txt";
    }
    static final class SearchStats {
        long exploredStates = 0;
        long emittedPathCandidates = 0;
        boolean truncatedByPathLimit = false;
        boolean truncatedByStateLimit = false;
        boolean stopRequested = false;
    }

    private static final class PathWriteResult {
        int totalPaths = 0;
        long exploredStates = 0;
        long emittedPathCandidates = 0;
        boolean truncatedByPathLimit = false;
        boolean truncatedByStateLimit = false;
    }

    private static final class PathCandidate {
        final String rendered;
        final int score;

        PathCandidate(String rendered, int score) {
            this.rendered = rendered;
            this.score = score;
        }
    }

    private String projectPackagePrefix = ""; 
    private String projectRootPrefix = "";  

    public static void main(String[] args) throws Exception {
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "error");
        System.setProperty("org.slf4j.simpleLogger.log.sootup", "error");
        System.setProperty("org.slf4j.simpleLogger.showDateTime", "false");

        CliConfig cfg = parseCliArgs(args);
        if (cfg == null) return;

        new CallGraphPathFinder().run(
                cfg.classPath,
                cfg.targetClass,
                cfg.targetMethodName,
                cfg.maxDepth,
                cfg.outputFile,
                cfg.mode,
                cfg.limitPackageRoots,
                cfg.depDir,
                cfg.outputJson,
                cfg.maxPaths,
                cfg.maxSearchStates,
                cfg.threads);
    }

    static CliConfig parseCliArgs(String[] args) {
        if (args.length < 3) {
            System.err.println(USAGE);
            return null;
        }

        CliConfig cfg = new CliConfig();
        int index = 0;

        while (index < args.length && args[index].startsWith("--")) {
            String opt = args[index];
            if (opt.startsWith("--roots=")) {
                String v = opt.substring("--roots=".length()).toUpperCase(Locale.ROOT);
                try { cfg.mode = RootMode.valueOf(v); } catch (IllegalArgumentException e) { System.err.println("Unknown root mode " + v + ", defaulting to PUBLIC"); }
                index++;
            } else if (opt.startsWith("--limitPackageRoots=")) {
                String v = opt.substring("--limitPackageRoots=".length()).toLowerCase(Locale.ROOT);
                cfg.limitPackageRoots = !("false".equals(v) || "0".equals(v) || "no".equals(v));
                index++;
            } else if (opt.startsWith("--depDir=")) {
                cfg.depDir = opt.substring("--depDir=".length());
                index++;
            } else if (opt.startsWith("--outputJson=")) {
                cfg.outputJson = opt.substring("--outputJson=".length());
                index++;
            } else if (opt.startsWith("--maxPaths=")) {
                String v = opt.substring("--maxPaths=".length());
                try {
                    cfg.maxPaths = Integer.parseInt(v);
                    if (cfg.maxPaths < 0) cfg.maxPaths = 0;
                } catch (NumberFormatException e) {
                    System.err.println("Invalid --maxPaths value '" + v + "', defaulting to 0 (unlimited)");
                    cfg.maxPaths = 0;
                }
                index++;
            } else if (opt.startsWith("--maxSearchStates=")) {
                String v = opt.substring("--maxSearchStates=".length());
                try {
                    cfg.maxSearchStates = Long.parseLong(v);
                    if (cfg.maxSearchStates < 0) cfg.maxSearchStates = 0;
                } catch (NumberFormatException e) {
                    System.err.println("Invalid --maxSearchStates value '" + v + "', defaulting to 0 (unlimited)");
                    cfg.maxSearchStates = 0;
                }
                index++;
            } else if (opt.startsWith("--threads=")) {
                String v = opt.substring("--threads=".length());
                try {
                    cfg.threads = Integer.parseInt(v);
                    if (cfg.threads < 1) {
                        System.err.println("Invalid --threads value '" + v + "', must be >= 1; defaulting to 1");
                        cfg.threads = 1;
                    }
                } catch (NumberFormatException e) {
                    System.err.println("Invalid --threads value '" + v + "', defaulting to 1");
                    cfg.threads = 1;
                }
                index++;
            } else {
                break;
            }
        }

        if (args.length - index < 3) {
            System.err.println(USAGE);
            return null;
        }
        cfg.classPath = args[index++];
        cfg.targetClass = args[index++];
        cfg.targetMethodName = args[index++];
        
        if (args.length - index >= 1) {
            String nextArg = args[index];
            try {
                cfg.maxDepth = Integer.parseInt(nextArg);
                index++;
                if (args.length - index >= 1) {
                    cfg.outputFile = args[index];
                }
            } catch (NumberFormatException e) {
                cfg.outputFile = nextArg;
            }
        }
        return cfg;
    }

    private void run(String classPath, String targetClassFQN, String targetMethodName, int maxDepth, String outputFile, RootMode rootMode, boolean limitPackageRoots, String depDir, String outputJson, int maxPaths, long maxSearchStates, int threads) throws IOException {
        long start = System.currentTimeMillis();
        System.out.println("Phase 1: Analyzing main classpath to identify external dependencies...");
        Set<String> externalClasses = new LinkedHashSet<>();
        String enhancedClassPath = classPath;
        
        if (depDir != null && !depDir.isEmpty()) {
            JavaView quickView = buildViewNoAutoDeps(classPath);
            Set<String> knownClasses = new HashSet<>();
            quickView.getClasses().forEach(sc -> knownClasses.add(sc.getClassSource().getClassType().getFullyQualifiedName()));
            
            for (var sc : quickView.getClasses().collect(Collectors.toList())) {
                for (var m : sc.getMethods()) {
                    String sub = m.getSubSignature().toString();
                    int l = sub.indexOf('(');
                    int r = sub.lastIndexOf(')');
                    if (l >= 0 && r > l) {
                        String params = sub.substring(l + 1, r).trim();
                        if (!params.isEmpty()) {
                            for (String param : params.split(",")) {
                                String pType = param.trim();
                                while (pType.endsWith("[]")) pType = pType.substring(0, pType.length() - 2).trim();
                                if (!isPrimitiveName(pType) && !knownClasses.contains(pType) && !pType.startsWith("java.lang.")) {
                                    externalClasses.add(pType);
                                }
                            }
                        }
                    }
                }
            }
            
            if (!externalClasses.isEmpty()) {
                System.out.println("Found " + externalClasses.size() + " external classes referenced:");
                for (String ext : externalClasses) {
                    System.out.println("  - " + ext);
                }
                Map<String, Set<String>> jarToClasses = scanJarsForClasses(depDir, externalClasses);
                
                if (!jarToClasses.isEmpty()) {
                    System.out.println("\nRequired dependency JARs:");
                    StringBuilder additionalCp = new StringBuilder();
                    for (Map.Entry<String, Set<String>> e : jarToClasses.entrySet()) {
                        System.out.println("  " + e.getKey() + " (provides: " + String.join(", ", e.getValue()) + ")");
                        if (additionalCp.length() > 0) additionalCp.append(File.pathSeparator);
                        additionalCp.append(e.getKey());
                    }
                    enhancedClassPath = classPath + File.pathSeparator + additionalCp;
                    System.out.println("\nPhase 2: Re-analyzing with required dependencies...");
                } else {
                    System.out.println("No JARs found containing the external classes in " + depDir);
                }
            }
        }
        
        // Derive project package prefix from the target class FQN so we can mark
        // third-party dependency JARs as SourceType.Library.  Library classes are
        // used by SootUp for type resolution but their method bodies are never
        // parsed during CHA, which prevents crashes on malformed third-party bytecode.
        String inferredProjectPrefix = targetClassFQN.contains(".")
                ? targetClassFQN.substring(0, targetClassFQN.lastIndexOf('.'))
                : "";
        JavaView view = buildViewWithSplit(enhancedClassPath, inferredProjectPrefix);
        String resolvedTargetClassFQN = resolveClassFQN(view, targetClassFQN, targetMethodName);
        final var classesListAll = view.getClasses().collect(Collectors.toList());
        for (var sc : classesListAll) {
            String classFqn = sc.getClassSource().getClassType().getFullyQualifiedName();
            try {
                String cvis;
                if (sc.isPublic()) cvis = "public";
                else if (sc.isProtected()) cvis = "protected";
                else if (sc.isPrivate()) cvis = "private";
                else cvis = "default";
                boolean isAbs = false;
                boolean isFinal = false;
                try { isAbs = sc.isAbstract(); } catch (Throwable ignored) {}
                try { isFinal = sc.isFinal(); } catch (Throwable ignored) {}
                StringBuilder mods = new StringBuilder(cvis);
                if (isFinal) mods.append(" final");
                if (isAbs) mods.append(" abstract");
                classModifiers.put(classFqn, mods.toString());
                if (sc.isPublic()) {
                    try {
                        //superclass
                        var superOpt = sc.getSuperclass();
                        if (superOpt.isPresent()) {
                            String superFqn = superOpt.get().getFullyQualifiedName();
                            if (!superFqn.equals("java.lang.Object")) {
                                implementationsOf.computeIfAbsent(superFqn, k -> new ArrayList<>()).add(classFqn);
                            }
                        }
                        for (var iface : sc.getInterfaces()) {
                            String ifaceFqn = iface.getFullyQualifiedName();
                            implementationsOf.computeIfAbsent(ifaceFqn, k -> new ArrayList<>()).add(classFqn);
                        }
                    } catch (Throwable ignored) {}
                }
            } catch (Throwable t) {
                classModifiers.put(classFqn, "");
            }
            for (var m : sc.getMethods()) {
                String vis;
                if (m.isPublic()) vis = "public"; else if (m.isProtected()) vis = "protected"; else if (m.isPrivate()) vis = "private"; else vis = "default";
                methodModifiers.put(buildKey(m.getSignature()), vis);
                if ("<init>".equals(m.getName())) {
                    String sub = m.getSubSignature().toString();
                    int l = sub.indexOf('(');
                    int r = sub.lastIndexOf(')');
                    String params = (l >= 0 && r >= 0 && r > l) ? sub.substring(l + 1, r).trim() : "";
                    constructorsByClass.computeIfAbsent(classFqn, k -> new ArrayList<>()).add(params);
                    constructorModsByClass.computeIfAbsent(classFqn, k -> new ArrayList<>()).add(vis);
                }
                if (m.isPublic() && m.isStatic() && !m.getName().startsWith("<")) {
                    try {
                        String sub = m.getSubSignature().toString();
                        int spaceIdx = sub.indexOf(' ');
                        if (spaceIdx > 0) {
                            String returnType = sub.substring(0, spaceIdx);
                            if (!returnType.equals("void") && !isPrimitiveName(returnType)) {
                                String baseReturn = returnType;
                                while (baseReturn.endsWith("[]")) baseReturn = baseReturn.substring(0, baseReturn.length() - 2);
                                if (!isPrimitiveName(baseReturn)) {
                                    String methodDesc = classFqn + "#" + sub.substring(spaceIdx + 1) + ": " + returnType;
                                    factoryMethodsFor.computeIfAbsent(baseReturn, k -> new ArrayList<>()).add(methodDesc);
                                }
                            }
                        }
                    } catch (Throwable ignored) {}
                }
            }
        }

    String targetPkgPrefix = "";
    int lastDotIdx = resolvedTargetClassFQN.lastIndexOf('.');
    if (lastDotIdx > 0) targetPkgPrefix = resolvedTargetClassFQN.substring(0, lastDotIdx);
    this.projectPackagePrefix = targetPkgPrefix;
    this.projectRootPrefix = computeProjectRootPrefix(resolvedTargetClassFQN);
    List<MethodSignature> entryPoints = selectRoots(view, rootMode, limitPackageRoots ? targetPkgPrefix : "");
        if (entryPoints.isEmpty()) {
            System.err.println("No entry points discovered under mode=" + rootMode + ". Aborting.");
            return;
        }

        CallGraphAlgorithm cha = new ClassHierarchyAnalysisAlgorithm(view);
        CallGraph cg = cha.initialize(entryPoints);

    Map<MethodSignature, Set<MethodSignature>> forward = new HashMap<>();
    cg.getCalls().forEach(call -> forward.computeIfAbsent(call.sourceMethodSignature(), k -> new LinkedHashSet<>()).add(call.targetMethodSignature()));

        List<MethodSignature> targets = new ArrayList<>();
        final var classesList = view.getClasses().collect(Collectors.toList());
        for (var sc : classesList) {
            String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
            if (fqn.equals(resolvedTargetClassFQN)) {
                for (var m : sc.getMethods()) {
                    if (m.getName().equals(targetMethodName)) {
                        targets.add(m.getSignature());
                    }
                }
            }
        }
        if (targets.isEmpty()) {
            System.err.println("No target methods resolved for " + resolvedTargetClassFQN + "." + targetMethodName);
            return;
        }

        int augmentedPrivateEdges = augmentIntraClassEdgesForPrivateTargets(
            classesList,
                forward,
                resolvedTargetClassFQN,
            targetMethodName,
                new HashSet<>(targets));
        if (augmentedPrivateEdges > 0) {
            System.out.println("Augmented private-target intra-class edges: " + augmentedPrivateEdges);
        }

        Set<MethodSignature> canReachTarget = new HashSet<>(targets);
        boolean changed;
        do {
            changed = false;
            for (Map.Entry<MethodSignature, Set<MethodSignature>> e : forward.entrySet()) {
                if (!canReachTarget.contains(e.getKey())) {
                    for (MethodSignature t : e.getValue()) {
                        if (canReachTarget.contains(t)) {
                            canReachTarget.add(e.getKey());
                            changed = true;
                            break;
                        }
                    }
                }
            }
        } while (changed);

        Map<MethodSignature, Integer> minDistanceToTarget = computeMinDistanceToTargets(forward, new HashSet<>(targets), maxDepth);
        PathWriteResult result = writePathsIterative(
                outputFile,
                outputJson,
                forward,
                entryPoints,
                canReachTarget,
                new HashSet<>(targets),
                rootMode,
                resolvedTargetClassFQN,
                targetMethodName,
                maxDepth,
                limitPackageRoots,
                minDistanceToTarget,
                maxPaths,
                maxSearchStates,
                threads);
        long elapsed = System.currentTimeMillis() - start;
        System.out.printf("Found %d paths (depth<=%d) ending in target(s) using roots=%s in %d ms. Target class: %s Output: %s%n", result.totalPaths, maxDepth, rootMode, elapsed, resolvedTargetClassFQN, outputFile);
        if (result.truncatedByPathLimit || result.truncatedByStateLimit) {
            System.out.printf(
                    "Search truncated: byPathLimit=%s byStateLimit=%s exploredStates=%d emittedPathCandidates=%d%n",
                    result.truncatedByPathLimit,
                    result.truncatedByStateLimit,
                    result.exploredStates,
                    result.emittedPathCandidates);
        }
        if (outputJson != null && !outputJson.isEmpty()) {
            System.out.printf("JSON summary written to: %s%n", outputJson);
        }
    }

    private Map<MethodSignature, Integer> computeMinDistanceToTargets(
            Map<MethodSignature, Set<MethodSignature>> forward,
            Set<MethodSignature> targets,
            int maxDepth) {
        Map<MethodSignature, Set<MethodSignature>> reverse = new HashMap<>();
        for (Map.Entry<MethodSignature, Set<MethodSignature>> e : forward.entrySet()) {
            MethodSignature src = e.getKey();
            for (MethodSignature dst : e.getValue()) {
                reverse.computeIfAbsent(dst, k -> new LinkedHashSet<>()).add(src);
            }
        }

        Map<MethodSignature, Integer> dist = new HashMap<>();
        ArrayDeque<MethodSignature> queue = new ArrayDeque<>();
        for (MethodSignature t : targets) {
            dist.put(t, 0);
            queue.add(t);
        }

        while (!queue.isEmpty()) {
            MethodSignature cur = queue.removeFirst();
            int d = dist.get(cur);
            if (d >= maxDepth) continue;
            for (MethodSignature pred : reverse.getOrDefault(cur, Collections.emptySet())) {
                int nd = d + 1;
                Integer old = dist.get(pred);
                if (old == null || nd < old) {
                    dist.put(pred, nd);
                    queue.addLast(pred);
                }
            }
        }
        return dist;
    }

    private int augmentIntraClassEdgesForPrivateTargets(
            List<? extends SootClass> classes,
            Map<MethodSignature, Set<MethodSignature>> forward,
            String targetClassFQN,
            String targetMethodName,
            Set<MethodSignature> targets) {
        if (targets.isEmpty()) return 0;

        List<SootMethod> classMethods = new ArrayList<>();
        for (SootClass sc : classes) {
            String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
            if (!fqn.equals(targetClassFQN)) continue;
            for (var m : sc.getMethods()) {
                classMethods.add(m);
            }
            break;
        }

        if (classMethods.isEmpty()) return 0;

        boolean hasPrivateTarget = false;
        Set<MethodSignature> sameClassSignatures = new HashSet<>();
        Map<String, List<MethodSignature>> sameClassBySubSignature = new HashMap<>();
        Map<String, List<MethodSignature>> sameClassByNameAndArity = new HashMap<>();
        for (SootMethod method : classMethods) {
            MethodSignature signature = method.getSignature();
            sameClassSignatures.add(signature);
            sameClassBySubSignature
                    .computeIfAbsent(signature.getSubSignature().toString(), k -> new ArrayList<>())
                    .add(signature);
            sameClassByNameAndArity
                    .computeIfAbsent(method.getName() + "/" + method.getParameterCount(), k -> new ArrayList<>())
                    .add(signature);
            if ((targets.contains(signature) || method.getName().equals(targetMethodName))
                    && !method.isPublic()
                    && !method.isProtected()) {
                hasPrivateTarget = true;
            }
        }
        if (!hasPrivateTarget) return 0;

        int added = 0;
        for (SootMethod method : classMethods) {
            if (!method.hasBody()) continue;
            MethodSignature source = method.getSignature();
            try {
                for (var stmt : method.getBody().getStmts()) {
                    if (!stmt.isInvokableStmt()) continue;
                    Optional<AbstractInvokeExpr> invokeExpr = stmt.asInvokableStmt().getInvokeExpr();
                    if (invokeExpr.isEmpty()) continue;
                    MethodSignature invoked = invokeExpr.get().getMethodSignature();
                    List<MethodSignature> targetsToAdd = new ArrayList<>();
                    if (sameClassSignatures.contains(invoked)) {
                        targetsToAdd.add(invoked);
                    } else if (sameClassBySubSignature.containsKey(invoked.getSubSignature().toString())) {
                        targetsToAdd.addAll(sameClassBySubSignature.get(invoked.getSubSignature().toString()));
                    } else {
                        String key = invoked.getName() + "/" + invoked.getParameterTypes().size();
                        targetsToAdd.addAll(sameClassByNameAndArity.getOrDefault(key, Collections.emptyList()));
                    }
                    Set<MethodSignature> edges = forward.computeIfAbsent(source, k -> new LinkedHashSet<>());
                    for (MethodSignature target : targetsToAdd) {
                        if (edges.add(target)) {
                            added++;
                        }
                    }
                }
            } catch (Throwable ignored) {
                // Keep SootUp's public-method behavior unchanged if a private body cannot be inspected.
            }
        }
        return added;
    }

    private String resolveClassFQN(JavaView view, String userProvided, String targetMethodName) {
        if (userProvided.contains(".")) {
            return userProvided; 
        }
        String simple = userProvided;
        List<String> matches = new ArrayList<>();
        final var classesList = view.getClasses().collect(Collectors.toList());
        for (var sc : classesList) {
            String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
            if (simpleName(fqn).equals(simple)) {
                matches.add(fqn);
            }
        }
        if (matches.isEmpty()) {
            System.err.println("Could not resolve simple class name '" + simple + "' on classpath. Proceeding with original value (may fail): " + userProvided);
            return userProvided;
        }
        if (matches.size() > 1 && targetMethodName != null && !targetMethodName.isEmpty()) {
            List<String> narrowed = new ArrayList<>();
            for (var sc : classesList) {
                String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
                if (matches.contains(fqn)) {
                    boolean hasMethod = false;
                    for (var m : sc.getMethods()) {
                        if (m.getName().equals(targetMethodName)) { hasMethod = true; break; }
                    }
                    if (hasMethod) narrowed.add(fqn);
                }
            }
            if (!narrowed.isEmpty()) {
                matches = narrowed;
            }
        }
        if (matches.size() > 1) {
            System.err.println("Ambiguous simple class name '" + simple + "' resolved to multiple FQNs: " + matches + ". Using first: " + matches.get(0));
        }
        return matches.get(0);
    }

    private String simpleName(String fqn) {
        int lastDot = fqn.lastIndexOf('.');
        return lastDot >= 0 ? fqn.substring(lastDot + 1) : fqn;
    }

    private List<MethodSignature> selectRoots(JavaView view, RootMode mode, String packagePrefix) {
        List<MethodSignature> roots = new ArrayList<>();
        final String pkg = packagePrefix == null ? "" : packagePrefix;
        final var classesList = view.getClasses().collect(Collectors.toList());
        switch (mode) {
            case MAINS -> {
                for (var sc : classesList) {
                    String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
                    if (!pkg.isEmpty() && !fqn.startsWith(pkg)) continue;
                    for (var m : sc.getMethods()) {
                        if (m.isPublic() && m.getName().equals("main")) roots.add(m.getSignature());
                    }
                }
            }
            case PUBLIC -> {
                for (var sc : classesList) {
                    String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
                    if (!pkg.isEmpty() && !fqn.startsWith(pkg)) continue;
                    for (var m : sc.getMethods()) {
                        if (m.isPublic()) roots.add(m.getSignature());
                    }
                }
            }
            case ALL -> {
                for (var sc : classesList) {
                    String fqn = sc.getClassSource().getClassType().getFullyQualifiedName();
                    if (!pkg.isEmpty() && !fqn.startsWith(pkg)) continue;
                    for (var m : sc.getMethods()) {
                        roots.add(m.getSignature());
                    }
                }
            }
        }
        return roots;
    }

    private JavaView buildViewNoAutoDeps(String classPath) {
        List<AnalysisInputLocation> inputs = new ArrayList<>();
        Set<String> added = new LinkedHashSet<>();

        for (Path entry : resolveClasspathEntries(classPath)) {
            addInput(entry.toString(), inputs, added);
        }
        inputs.add(new DefaultRuntimeAnalysisInputLocation());
        System.out.println("Resolved classpath entries (" + added.size() + "): ");
        for (String s : added) System.out.println("  " + s);
        return new JavaView(inputs);
    }

    private Map<String, Set<String>> scanJarsForClasses(String depDir, Set<String> targetClasses) {
        Map<String, Set<String>> result = new LinkedHashMap<>();
        Path depPath = Paths.get(depDir);

        if (!Files.isDirectory(depPath)) {
            System.err.println("Dependency directory not found: " + depDir);
            return result;
        }
        Map<String, String> classToPath = new HashMap<>();
        Map<String, String> pathToClass = new HashMap<>();
        for (String cls : targetClasses) {
            String path = cls.replace('.', '/') + ".class";
            classToPath.put(cls, path);
            pathToClass.put(path, cls);
        }
        Set<String> foundClasses = new HashSet<>();
        Set<String> targetPaths = new HashSet<>(pathToClass.keySet());

        try (Stream<Path> walk = Files.walk(depPath, 2)) {
            List<Path> jars = walk.filter(p -> p.toString().endsWith(".jar")).collect(Collectors.toList());

            for (Path jarPath : jars) {
                if (foundClasses.size() == targetClasses.size()) {
                    break; 
                }
                Set<String> providedClasses = new LinkedHashSet<>();
                try (JarFile jar = new JarFile(jarPath.toFile())) {
                    Enumeration<JarEntry> entries = jar.entries();
                    while (entries.hasMoreElements()) {
                        JarEntry entry = entries.nextElement();
                        String entryName = entry.getName();
                        if (!entryName.endsWith(".class")) {
                            continue;
                        }
                        if (targetPaths.contains(entryName)) {
                            String cls = pathToClass.get(entryName);
                            if (cls != null && !foundClasses.contains(cls)) {
                                providedClasses.add(cls);
                                foundClasses.add(cls);
                            }
                        }
                    }
                } catch (IOException e) {
                    System.err.println("Warning: Could not read JAR " + jarPath + ": " + e.getMessage());
                }

                if (!providedClasses.isEmpty()) {
                    result.put(jarPath.toString(), providedClasses);
                }
            }
        } catch (IOException e) {
            System.err.println("Error scanning dependency directory: " + e.getMessage());
        }

        Set<String> notFound = new LinkedHashSet<>(targetClasses);
        notFound.removeAll(foundClasses);
        if (!notFound.isEmpty()) {
            System.out.println("Classes not found in any JAR (may be JDK classes or missing):");
            for (String cls : notFound) {
                System.out.println("  - " + cls);
            }
        }

        return result;
    }

    private void addInput(String path, List<AnalysisInputLocation> inputs, Set<String> added) {
        if (added.contains(path)) return;
        inputs.add(new JavaClassPathAnalysisInputLocation(path));
        added.add(path);
    }

    private void addInputTyped(String path, SourceType srcType, List<AnalysisInputLocation> inputs, Set<String> added) {
        if (added.contains(path)) return;
        inputs.add(new JavaClassPathAnalysisInputLocation(path, srcType));
        added.add(path);
    }

    /**
     * Returns true if the JAR (or directory) contains at least one class whose
     * package starts with {@code prefix}.
     */
    private boolean jarContainsPrefix(String path, String prefix) {
        if (prefix == null || prefix.isEmpty()) return true;
        String pkgPath = prefix.replace('.', '/');
        Path p = Paths.get(path);
        if (Files.isDirectory(p)) {
            try (Stream<Path> walk = Files.walk(p)) {
                return walk.anyMatch(f -> {
                    String s = f.toString().replace('\\', '/');
                    return s.contains("/" + pkgPath + "/") && s.endsWith(".class");
                });
            } catch (IOException e) { return false; }
        }
        if (!path.endsWith(".jar")) return true;
        try (JarFile jar = new JarFile(path)) {
            Enumeration<JarEntry> entries = jar.entries();
            while (entries.hasMoreElements()) {
                String name = entries.nextElement().getName();
                if (name.startsWith(pkgPath + "/") && name.endsWith(".class")) return true;
            }
        } catch (IOException e) { /* ignore */ }
        return false;
    }

    /**
     * Builds a view where classpath entries that contain project classes
     * (matching {@code projectPrefix}) are loaded as {@link SourceType#Application}
     * and all other entries are loaded as {@link SourceType#Library}.
     * <p>
     * This prevents SootUp's CHA from trying to parse method bodies in
     * third-party JARs (e.g. okhttp3), which can crash on unusual bytecode.
     */
    private JavaView buildViewWithSplit(String classPath, String projectPrefix) {
        List<AnalysisInputLocation> inputs = new ArrayList<>();
        Set<String> added = new LinkedHashSet<>();
        List<Path> resolvedEntries = resolveClasspathEntries(classPath);

        // First pass: determine which concrete paths are "project" entries.
        Set<String> appPaths = new LinkedHashSet<>();
        for (Path entry : resolvedEntries) {
            if (jarContainsPrefix(entry.toString(), projectPrefix)) {
                appPaths.add(entry.toString());
            }
        }

        // Second pass: add inputs with correct SourceType.
        for (Path entry : resolvedEntries) {
            SourceType st = appPaths.contains(entry.toString()) ? SourceType.Application : SourceType.Library;
            addInputTyped(entry.toString(), st, inputs, added);
        }

        inputs.add(new DefaultRuntimeAnalysisInputLocation());
        System.out.println("Resolved classpath entries (" + added.size() + ") [" + appPaths.size() + " application, " + (added.size() - appPaths.size()) + " library]: ");
        for (String s : added) System.out.println("  " + (appPaths.contains(s) ? "[APP] " : "[LIB] ") + s);
        return new JavaView(inputs);
    }

    private List<Path> resolveClasspathEntries(String classPath) {
        // Normalize plain and wildcard classpath entries into concrete existing paths.
        List<Path> resolved = new ArrayList<>();
        String separator = Pattern.quote(File.pathSeparator);
        for (String rawEntry : classPath.split(separator)) {
            if (rawEntry == null || rawEntry.trim().isEmpty()) continue;
            String entry = rawEntry.trim();
            if (entry.contains("*")) {
                expandWildcard(entry).forEach(resolved::add);
                continue;
            }
            Path path = Paths.get(entry);
            if (!Files.exists(path)) {
                System.err.println("Warning: classpath entry not found: " + entry);
                continue;
            }
            resolved.add(path);
        }
        return resolved;
    }

    private List<Path> expandWildcard(String pattern) {
        List<Path> out = new ArrayList<>();
        try {
            if (pattern.endsWith("/*")) {
                Path dir = Paths.get(pattern.substring(0, pattern.length() - 2));
                if (Files.isDirectory(dir)) {
                    try (Stream<Path> stream = Files.list(dir)) {
                        stream.filter(p -> p.toString().endsWith(".jar") || Files.isDirectory(p))
                              .forEach(out::add);
                    }
                }
            } else if (pattern.contains("**")) {
                String baseStr = pattern.substring(0, pattern.indexOf("**"));
                Path base = Paths.get(baseStr);
                if (Files.isDirectory(base)) {
                    try (Stream<Path> walk = Files.walk(base, 5)) {
                        walk.filter(p -> p.toString().endsWith(".jar")).forEach(out::add);
                    }
                }
            } else {
                Path p = Paths.get(pattern);
                if (Files.exists(p)) out.add(p);
            }
        } catch (IOException ignored) {
        }
        return out;
    }

    private PathWriteResult writePathsIterative(String file,
                                    String outputJson,
                                    Map<MethodSignature, Set<MethodSignature>> forward,
                                    List<MethodSignature> roots,
                                    Set<MethodSignature> canReachTarget,
                                    Set<MethodSignature> targets,
                                    RootMode mode,
                                    String targetClass,
                                    String targetMethod,
                                    int maxDepth,
                                    boolean limitPackageRoots,
                                    Map<MethodSignature, Integer> minDistanceToTarget,
                                    int maxPaths,
                                    long maxSearchStates,
                                    int threads) throws IOException {
        PathWriteResult result = new PathWriteResult();
        Set<String> seenPathOutputs = Collections.synchronizedSet(new HashSet<>());
        Set<String> allInitClasses = Collections.synchronizedSet(new LinkedHashSet<>());
        
        allInitClasses.add(targetClass);
        
        List<PathCandidate> uniquePathCandidates = Collections.synchronizedList(new ArrayList<>());
        SearchStats searchStats = new SearchStats();
        Object pathLock = new Object();
        int candidateLimit = maxPaths > 0 ? Math.max(maxPaths, maxPaths * 4) : 0;
        
        Consumer<Deque<MethodSignature>> onPathFound = stack -> {
            String pathStr = formatSinglePathAndCollectInits(stack, allInitClasses, targets);
            if (!pathStr.isEmpty() && !seenPathOutputs.contains(pathStr)) {
                synchronized (pathLock) {
                    if (candidateLimit > 0 && uniquePathCandidates.size() >= candidateLimit) {
                        searchStats.truncatedByPathLimit = true;
                        searchStats.stopRequested = true;
                        return;
                    }
                    if (!seenPathOutputs.contains(pathStr)) {
                        uniquePathCandidates.add(new PathCandidate(pathStr, scorePath(stack, targetClass, targets)));
                        seenPathOutputs.add(pathStr);
                        if (candidateLimit > 0 && uniquePathCandidates.size() >= candidateLimit) {
                            searchStats.truncatedByPathLimit = true;
                            searchStats.stopRequested = true;
                        }
                    }
                }
            }
        };

        // Execute root searches in parallel (or sequentially if threads == 1)
        ParallelSearchManager parallelMgr = new ParallelSearchManager(threads);
        try {
            List<ParallelSearchManager.SearchTask> tasks = new ArrayList<>();
            
                List<MethodSignature> sortedRoots = new ArrayList<>(roots);
                sortedRoots.sort(Comparator
                    .comparingInt((MethodSignature root) -> root.getDeclClassType().getFullyQualifiedName().equals(targetClass) ? 0 : 1)
                    .thenComparingInt(root -> visibilityRank(visibilityOf(root)))
                    .thenComparing(MethodSignature::toString));
                for (MethodSignature root : sortedRoots) {
                if (!canReachTarget.contains(root)) continue;
                Integer dist = minDistanceToTarget.get(root);
                if (dist == null || dist > maxDepth) continue;
                
                SearchStats taskStats = new SearchStats();
                Runnable taskLogic = () -> {
                    findAllPathsUpToDepth(
                            forward,
                            root,
                            targets,
                            canReachTarget,
                            minDistanceToTarget,
                            maxDepth,
                            0,
                            new ArrayDeque<>(),
                            new HashSet<>(),
                            onPathFound,
                            taskStats,
                            maxSearchStates);
                    // Copy task stats to global (this will be aggregated by manager)
                    synchronized (pathLock) {
                        searchStats.exploredStates += taskStats.exploredStates;
                        searchStats.emittedPathCandidates += taskStats.emittedPathCandidates;
                        searchStats.truncatedByPathLimit |= taskStats.truncatedByPathLimit;
                        searchStats.truncatedByStateLimit |= taskStats.truncatedByStateLimit;
                        searchStats.stopRequested |= taskStats.stopRequested;
                    }
                };
                tasks.add(new ParallelSearchManager.SearchTask(taskLogic, taskStats));
            }
            
            try {
                parallelMgr.executeParallel(tasks, searchStats);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.println("Parallel search was interrupted: " + e.getMessage());
            }
        } finally {
            parallelMgr.shutdown();
        }
        
        if (uniquePathCandidates.isEmpty()) {
            for (MethodSignature t : targets) {
                if (t.getDeclClassType().getFullyQualifiedName().equals(targetClass)
                    && t.getName().equals(targetMethod)) {
                    ArrayDeque<MethodSignature> self = new ArrayDeque<>();
                    self.addLast(t);
                    String pathStr = formatSinglePathAndCollectInits(self, allInitClasses, targets);
                    if (!pathStr.isEmpty() && !seenPathOutputs.contains(pathStr)) {
                        uniquePathCandidates.add(new PathCandidate(pathStr, scorePath(self, targetClass, targets)));
                        seenPathOutputs.add(pathStr);
                    }
                }
            }
        }
        List<PathCandidate> sortedPathCandidates = new ArrayList<>(uniquePathCandidates);
        sortedPathCandidates.sort(Comparator
                .comparingInt((PathCandidate p) -> p.score)
                .thenComparing(p -> p.rendered));
        if (maxPaths > 0 && sortedPathCandidates.size() > maxPaths) {
            sortedPathCandidates = new ArrayList<>(sortedPathCandidates.subList(0, maxPaths));
            searchStats.truncatedByPathLimit = true;
        }
        List<String> uniquePathStrings = sortedPathCandidates.stream()
                .map(p -> p.rendered)
                .collect(Collectors.toList());
        result.totalPaths = uniquePathStrings.size();
        result.exploredStates = searchStats.exploredStates;
        result.emittedPathCandidates = searchStats.emittedPathCandidates;
        result.truncatedByPathLimit = searchStats.truncatedByPathLimit;
        result.truncatedByStateLimit = searchStats.truncatedByStateLimit;

        List<String> initLines;
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            bw.write("# Target: " + targetClass + "." + targetMethod + "\n");
            bw.write("# Roots: " + mode + ", MaxDepth: " + maxDepth + ", LimitPackageRoots: " + limitPackageRoots + "\n");
            bw.write("# SearchLimits: maxPaths=" + maxPaths + ", maxSearchStates=" + maxSearchStates + "\n");
            bw.write("# SearchStats: exploredStates=" + result.exploredStates + ", emittedPathCandidates=" + result.emittedPathCandidates + ", truncatedByPathLimit=" + result.truncatedByPathLimit + ", truncatedByStateLimit=" + result.truncatedByStateLimit + "\n");
            bw.flush();

            if (!uniquePathStrings.isEmpty()) {
                bw.write("## Max Depth Paths\n");
                for (String p : uniquePathStrings) bw.write(p);
                bw.flush();
            } else {
                bw.write("## No Public Paths Found\n");
                bw.write("# Note: No public methods in public classes can reach the target method within maxDepth.\n");
                bw.write("# The target method may be non-public or only reachable via non-public paths.\n");
                bw.write("# Initialization information for the target class is provided below.\n\n");
                bw.flush();
            }
            LinkedHashSet<String> expandedInitClasses = new LinkedHashSet<>(allInitClasses);
            boolean foundNew;
            do {
                foundNew = false;
                LinkedHashSet<String> toAdd = new LinkedHashSet<>();
                for (String cls : expandedInitClasses) {
                    List<String> ctors = constructorsByClass.getOrDefault(cls, Collections.emptyList());
                    List<String> ctorMods = constructorModsByClass.getOrDefault(cls, Collections.emptyList());
                    for (int i = 0; i < ctors.size(); i++) {
                        String mod = i < ctorMods.size() ? ctorMods.get(i) : "";
                        if (!"public".equals(mod)) continue; 
                        String paramsCsv = ctors.get(i);
                        for (String pType : extractParamTypes(paramsCsv)) {
                            if (expandedInitClasses.contains(pType) || toAdd.contains(pType)) continue;
                            toAdd.add(pType);
                            foundNew = true;
                        }
                    }
                }
                expandedInitClasses.addAll(toAdd);
            } while (foundNew);
            
            bw.write("\n## All Initializations\n");
            initLines = buildInitializationLines(expandedInitClasses);
            if (initLines.isEmpty()) {
                bw.write("# No public class initializations found for target class.\n");
            } else {
                for (String line : initLines) {
                    bw.write(line);
                    bw.write("\n");
                }
            }
            bw.flush();
        }

        if (outputJson != null && !outputJson.isEmpty()) {
            writeJsonSummary(outputJson, file, targetClass, targetMethod, mode, maxDepth, limitPackageRoots, uniquePathStrings, initLines, maxPaths, maxSearchStates, result);
        }
        return result;
    }

    private void writeJsonSummary(String jsonFile,
                                  String textFile,
                                  String targetClass,
                                  String targetMethod,
                                  RootMode mode,
                                  int maxDepth,
                                  boolean limitPackageRoots,
                                  List<String> pathBlocks,
                                  List<String> initLines,
                                  int maxPaths,
                                  long maxSearchStates,
                                  PathWriteResult result) throws IOException {
        int indirect = 0;
        List<List<String>> stepLists = new ArrayList<>();
        for (String block : pathBlocks) {
            List<String> steps = parsePathSteps(block);
            stepLists.add(steps);
            if (steps.size() > 1) indirect++;
        }

        PathfindingResultSerializer.SerializationRequest req = new PathfindingResultSerializer.SerializationRequest();
        req.targetClass = targetClass;
        req.targetMethod = targetMethod;
        req.rootMode = mode.toString();
        req.maxDepth = maxDepth;
        req.limitPackageRoots = limitPackageRoots;
        req.maxPaths = maxPaths;
        req.maxSearchStates = maxSearchStates;
        req.pathCount = pathBlocks.size();
        req.indirectPathCount = indirect;
        req.initializationLineCount = initLines.size();
        req.exploredStates = result.exploredStates;
        req.emittedPathCandidates = result.emittedPathCandidates;
        req.truncatedByPathLimit = result.truncatedByPathLimit;
        req.truncatedByStateLimit = result.truncatedByStateLimit;
        req.textOutputFile = textFile;
        req.pathSteps = stepLists;
        req.initializationLines = new HashSet<>(initLines);

        PathfindingResultSerializer.serializeToJson(req, jsonFile);
    }

    private List<String> parsePathSteps(String pathBlock) {
        List<String> steps = new ArrayList<>();
        if (pathBlock == null || pathBlock.isEmpty()) return steps;
        for (String line : pathBlock.split("\\n")) {
            String step = line.trim();
            if (step.isEmpty() || "---".equals(step) || "Path:".equals(step)) continue;
            if (!line.startsWith("  ")) continue;
            if (step.contains("#") && step.contains("):")) {
                steps.add(step);
            }
        }
        return steps;
    }

    private boolean findAllPathsUpToDepth(Map<MethodSignature, Set<MethodSignature>> forward,
                                       MethodSignature current,
                                       Set<MethodSignature> targets,
                                       Set<MethodSignature> canReachTarget,
                                       Map<MethodSignature, Integer> minDistanceToTarget,
                                       int depthLimit,
                                       int currentDepth,
                                       Deque<MethodSignature> stack,
                                       Set<MethodSignature> onPath,
                                       Consumer<Deque<MethodSignature>> onPathFound,
                                       SearchStats searchStats,
                                       long maxSearchStates) {
        if (searchStats.stopRequested) return false;
        if (maxSearchStates > 0 && searchStats.exploredStates >= maxSearchStates) {
            searchStats.truncatedByStateLimit = true;
            searchStats.stopRequested = true;
            return false;
        }
        searchStats.exploredStates++;
        stack.addLast(current);
        onPath.add(current);
        if (targets.contains(current) && currentDepth <= depthLimit) {
            searchStats.emittedPathCandidates++;
            onPathFound.accept(new ArrayDeque<>(stack));
            if (searchStats.stopRequested) {
                stack.removeLast();
                onPath.remove(current);
                return false;
            }
        }
        if (currentDepth < depthLimit) {
            int nextRemainingDepth = depthLimit - currentDepth - 1;
            for (MethodSignature succ : forward.getOrDefault(current, Collections.emptySet())) {
                if (onPath.contains(succ)) continue;
                if (canReachTarget != null && !canReachTarget.contains(succ)) continue;
                Integer dist = minDistanceToTarget.get(succ);
                if (dist == null || dist > nextRemainingDepth) continue;
                boolean keepGoing = findAllPathsUpToDepth(
                        forward,
                        succ,
                        targets,
                        canReachTarget,
                        minDistanceToTarget,
                        depthLimit,
                        currentDepth + 1,
                        stack,
                        onPath,
                        onPathFound,
                        searchStats,
                        maxSearchStates);
                if (!keepGoing) {
                    stack.removeLast();
                    onPath.remove(current);
                    return false;
                }
            }
        }
        stack.removeLast();
        onPath.remove(current);
        return !searchStats.stopRequested;
    }

    private int scorePath(Deque<MethodSignature> stack, String targetClass, Set<MethodSignature> targets) {
        List<MethodSignature> steps = new ArrayList<>(stack);
        if (steps.isEmpty()) return Integer.MAX_VALUE;

        MethodSignature first = steps.get(0);
        int score = steps.size() * 100;
        if (!first.getDeclClassType().getFullyQualifiedName().equals(targetClass)) {
            score += 1000;
        }
        score += visibilityRank(visibilityOf(first)) * 100;
        if ("<init>".equals(first.getName()) || "<clinit>".equals(first.getName())) {
            score += 50;
        }

        boolean reachesTarget = false;
        for (MethodSignature step : steps) {
            if (targets.contains(step)) {
                reachesTarget = true;
                break;
            }
        }
        if (!reachesTarget) {
            score += 10000;
        }
        return score;
    }

    private int visibilityRank(String visibility) {
        if ("public".equals(visibility)) return 0;
        if ("protected".equals(visibility)) return 1;
        if ("default".equals(visibility)) return 2;
        if ("private".equals(visibility)) return 3;
        return 4;
    }

    private String formatSinglePathAndCollectInits(Deque<MethodSignature> stack, Set<String> allInitClasses, Set<MethodSignature> targets) {
        StringBuilder sb = new StringBuilder();
        LinkedHashSet<String> classesOnPath = new LinkedHashSet<>();
        boolean includeInternalSteps = false;
        for (MethodSignature target : targets) {
            if (!"public".equals(visibilityOf(target))) {
                includeInternalSteps = true;
                break;
            }
        }
        
        List<MethodSignature> methodsToPrint = new ArrayList<>();
        for (MethodSignature ms : stack) {
            String visibility = visibilityOf(ms);
            boolean isTarget = targets.contains(ms);
            
            if (isTarget) {
                classesOnPath.add(ms.getDeclClassType().getFullyQualifiedName());
                String subsig = ms.getSubSignature().toString();
                int lparen = subsig.indexOf('(');
                int rparen = subsig.lastIndexOf(')');
                if (lparen >= 0 && rparen > lparen) {
                    String params = subsig.substring(lparen + 1, rparen).trim();
                    if (!params.isEmpty()) {
                        for (String param : params.split(",")) {
                            String paramType = param.trim();
                            while (paramType.endsWith("[]")) {
                                paramType = paramType.substring(0, paramType.length() - 2).trim();
                            }
                            if (!isPrimitiveName(paramType)) {
                                classesOnPath.add(paramType);
                            }
                        }
                    }
                }
                if (!"public".equals(visibility)) {
                    if (includeInternalSteps) {
                        methodsToPrint.add(ms);
                    }
                    continue;
                }
            }
            if (includeInternalSteps && !"public".equals(visibility)) {
                methodsToPrint.add(ms);
                classesOnPath.add(ms.getDeclClassType().getFullyQualifiedName());
                continue;
            }
            if ("public".equals(visibility)) {
                String declClass = ms.getDeclClassType().getFullyQualifiedName();
                String classMod = classModifiers.getOrDefault(declClass, "").trim();
                if (!classMod.startsWith("public")) {
                    continue;
                }
                if (classMod.contains("abstract")) {
                    continue;
                }
                boolean hasUninstantiableParam = false;
                String subsig = ms.getSubSignature().toString();
                int lparen = subsig.indexOf('(');
                int rparen = subsig.lastIndexOf(')');
                if (lparen >= 0 && rparen > lparen) {
                    String params = subsig.substring(lparen + 1, rparen).trim();
                    if (!params.isEmpty()) {
                        for (String param : params.split(",")) {
                            String paramType = param.trim();
                            while (paramType.endsWith("[]")) {
                                paramType = paramType.substring(0, paramType.length() - 2).trim();
                            }
                            if (isPrimitiveName(paramType)) {
                                continue;
                            }
                            String paramMod = classModifiers.getOrDefault(paramType, "").trim();
                            if (paramMod.contains("abstract")) {
                                boolean hasUsableImpl = hasConcreteImplementation(paramType, new HashSet<>());
                                if (!hasUsableImpl) {
                                    hasUninstantiableParam = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (!hasUninstantiableParam) {
                    methodsToPrint.add(ms);
                    classesOnPath.add(ms.getDeclClassType().getFullyQualifiedName());
                    if (lparen >= 0 && rparen > lparen) {
                        String params = subsig.substring(lparen + 1, rparen).trim();
                        if (!params.isEmpty()) {
                            for (String param : params.split(",")) {
                                String paramType = param.trim();
                                while (paramType.endsWith("[]")) {
                                    paramType = paramType.substring(0, paramType.length() - 2).trim();
                                }
                                if (!isPrimitiveName(paramType)) {
                                    classesOnPath.add(paramType);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (methodsToPrint.isEmpty()) {
            return "";
        }
        LinkedHashSet<String> initClasses = new LinkedHashSet<>(classesOnPath);
        final int MAX_PARAM_CLOSURE_DEPTH = 2;
        String inProjectPrefix = (this.projectRootPrefix == null || this.projectRootPrefix.isEmpty())
            ? (this.projectPackagePrefix == null ? "" : this.projectPackagePrefix)
            : this.projectRootPrefix;
        Map<String, Integer> levelByClass = new HashMap<>();
        ArrayDeque<String> queue = new ArrayDeque<>();
        for (String c : initClasses) { levelByClass.put(c, 0); queue.add(c); }
        while (!queue.isEmpty()) {
            String cur = queue.pollFirst();
            int lvl = levelByClass.getOrDefault(cur, 0);
            if (lvl >= MAX_PARAM_CLOSURE_DEPTH) continue;
            List<String> ctors = constructorsByClass.getOrDefault(cur, Collections.emptyList());
            for (String paramsCsv : ctors) {
                for (String pType : extractParamTypes(paramsCsv)) {
                    if (!inProjectPrefix.isEmpty() && !pType.startsWith(inProjectPrefix)) continue;
                    if (!classModifiers.containsKey(pType) && !constructorsByClass.containsKey(pType)) continue;
                    if (initClasses.add(pType)) {
                        levelByClass.put(pType, lvl + 1);
                        queue.addLast(pType);
                    }
                }
            }
        }
        LinkedHashSet<String> publicInitClasses = new LinkedHashSet<>();
        for (String cls : initClasses) {
            String classMod = classModifiers.getOrDefault(cls, "").trim();
            if (classMod.startsWith("public")) {
                publicInitClasses.add(cls);
            }
        }
        allInitClasses.addAll(publicInitClasses);
        
        sb.append("Path:\n");
        for (MethodSignature ms : methodsToPrint) {
            sb.append("  ").append(formatSignature(ms)).append("\n");
        }
        sb.append("---\n");
        return sb.toString();
    }

    private List<String> buildInitializationLines(LinkedHashSet<String> initClasses) {
        List<String> initLines = new ArrayList<>();
        boolean firstPrinted = false;
        for (String cls : initClasses) {
            String classMod = classModifiers.getOrDefault(cls, "").trim();
            if (!classMod.isEmpty() && !classMod.startsWith("public")) {
                continue;
            }
            if (firstPrinted) {
                initLines.add("");
            }
            firstPrinted = true;

            List<String> ctors = constructorsByClass.getOrDefault(cls, Collections.emptyList());
            List<String> ctorMods = constructorModsByClass.getOrDefault(cls, Collections.emptyList());
            List<Integer> publicCtorIdxs = new ArrayList<>();
            for (int i = 0; i < ctors.size(); i++) {
                String mod = i < ctorMods.size() ? ctorMods.get(i) : "";
                if ("public".equals(mod)) publicCtorIdxs.add(i);
            }
            String modDisplay = classMod.isEmpty() ? "external" : classMod;
            initLines.add("  " + cls + " (" + modDisplay + "):");
            if (publicCtorIdxs.isEmpty()) {
                if (ctors.isEmpty() && classMod.isEmpty()) {
                    initLines.add("         external class - constructors unknown.");
                } else {
                    initLines.add("         no public constructors found.");
                }
            } else {
                for (int idx : publicCtorIdxs) {
                    String rawParams = ctors.get(idx);
                    String annotated = annotateParamModifiers(rawParams);
                    String simpleClassName = cls.contains(".") ? cls.substring(cls.lastIndexOf('.') + 1) : cls;
                    initLines.add("    " + simpleClassName + "(" + annotated + ") (public)");
                }
            }
            
            List<String> impls = implementationsOf.getOrDefault(cls, Collections.emptyList());
            if (!impls.isEmpty()) {
                Set<String> allImpls = new LinkedHashSet<>();
                collectAllImplementations(cls, allImpls, new HashSet<>());
                if (!allImpls.isEmpty()) {
                    initLines.add("    Known implementations:");
                    for (String impl : allImpls) {
                        String implMod = classModifiers.getOrDefault(impl, "").trim();
                        initLines.add("      - " + impl + (implMod.isEmpty() ? "" : " (" + implMod + ")"));
                    }
                }
            }
            
            List<String> factories = factoryMethodsFor.getOrDefault(cls, Collections.emptyList());
            if (!factories.isEmpty()) {
                initLines.add("    Factory methods:");
                for (String factory : factories) {
                    initLines.add("      - " + factory);
                }
            }
        }
        return initLines;
    }

    private void depthLimitedDfs(Map<MethodSignature, Set<MethodSignature>> forward,
                                 MethodSignature current,
                                 Set<MethodSignature> targets,
                                 int depthLimit,
                                 int currentDepth,
                                 Deque<MethodSignature> stack,
                                 Set<MethodSignature> onPath,
                                 List<String> out,
                                 Set<String> seenPathKeys) {
        stack.addLast(current);
        onPath.add(current);

        boolean isTarget = targets.contains(current);
        if (isTarget && currentDepth <= depthLimit) {
            if (currentDepth == depthLimit) {
                String key = buildPathKey(stack);
                if (!seenPathKeys.contains(key)) {
                    out.add(formatSinglePath(stack));
                    seenPathKeys.add(key);
                }
            }
        }
        if (currentDepth < depthLimit) {
            for (MethodSignature succ : forward.getOrDefault(current, Collections.emptySet())) {
                if (onPath.contains(succ)) continue;
                depthLimitedDfs(forward, succ, targets, depthLimit, currentDepth + 1, stack, onPath, out, seenPathKeys);
            }
        }
        stack.removeLast();
        onPath.remove(current);
    }

    private String formatSinglePath(Deque<MethodSignature> stack) {
        StringBuilder sb = new StringBuilder();
        LinkedHashSet<String> classesOnPath = new LinkedHashSet<>();
        for (MethodSignature ms : stack) {
            classesOnPath.add(ms.getDeclClassType().getFullyQualifiedName());
        }
        LinkedHashSet<String> initClasses = new LinkedHashSet<>(classesOnPath);
        final int MAX_PARAM_CLOSURE_DEPTH = 2;
    String inProjectPrefix = (this.projectRootPrefix == null || this.projectRootPrefix.isEmpty())
        ? (this.projectPackagePrefix == null ? "" : this.projectPackagePrefix)
        : this.projectRootPrefix;
        Map<String, Integer> levelByClass = new HashMap<>();
        ArrayDeque<String> queue = new ArrayDeque<>();
        for (String c : initClasses) { levelByClass.put(c, 0); queue.add(c); }
        while (!queue.isEmpty()) {
            String cur = queue.pollFirst();
            int lvl = levelByClass.getOrDefault(cur, 0);
            if (lvl >= MAX_PARAM_CLOSURE_DEPTH) continue;
            List<String> ctors = constructorsByClass.getOrDefault(cur, Collections.emptyList());
            for (String paramsCsv : ctors) {
                for (String pType : extractParamTypes(paramsCsv)) {
                    if (!inProjectPrefix.isEmpty() && !pType.startsWith(inProjectPrefix)) continue;
                    if (!classModifiers.containsKey(pType) && !constructorsByClass.containsKey(pType)) continue;
                    if (initClasses.add(pType)) {
                        levelByClass.put(pType, lvl + 1);
                        queue.addLast(pType);
                    }
                }
            }
        }
        List<String> initLines = new ArrayList<>();
        for (String cls : initClasses) {
            List<String> ctors = constructorsByClass.getOrDefault(cls, Collections.emptyList());
            List<String> ctorMods = constructorModsByClass.getOrDefault(cls, Collections.emptyList());
            if (!ctors.isEmpty()) {
                StringBuilder line = new StringBuilder();
                String classMod = classModifiers.getOrDefault(cls, "").trim();
                line.append("  ").append(cls);
                if (!classMod.isEmpty()) line.append(" (").append(classMod).append(")");
                line.append(": ");
                for (int i = 0; i < ctors.size(); i++) {
                    if (i > 0) line.append(", ");
                    String rawParams = ctors.get(i);
                    String annotated = annotateParamModifiers(rawParams);
                    line.append("(").append(annotated).append(")");
                    if (i < ctorMods.size()) {
                        String mod = ctorMods.get(i);
                        if (mod != null && !mod.isEmpty()) {
                            line.append(" (").append(mod).append(")");
                        }
                    }
                }
                line.append("\n");
                initLines.add(line.toString());
            }
        }
        if (!initLines.isEmpty()) {
            sb.append("Initializations:\n");
            for (String l : initLines) sb.append(l);
        }
        sb.append("Path:\n");
        for (MethodSignature ms : stack) {
            sb.append("  ").append(formatSignature(ms)).append("\n");
        }
        sb.append("---\n");
        return sb.toString();
    }

    private List<String> extractParamTypes(String paramsCsv) {
        if (paramsCsv == null || paramsCsv.isEmpty()) return Collections.emptyList();
        String[] parts = paramsCsv.split(",");
        List<String> out = new ArrayList<>(parts.length);
        for (String raw : parts) {
            String t = raw.trim();
            if (t.isEmpty()) continue;
            while (t.endsWith("[]")) t = t.substring(0, t.length() - 2);
            if (isPrimitiveName(t)) continue;
            out.add(t);
        }
        return out;
    }

    private boolean isPrimitiveName(String t) {
        return "boolean".equals(t) || "byte".equals(t) || "short".equals(t) || "char".equals(t)
                || "int".equals(t) || "long".equals(t) || "float".equals(t) || "double".equals(t);
    }

    private String buildPathKey(Deque<MethodSignature> stack) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (MethodSignature ms : stack) {
            if (!first) sb.append("->");
            first = false;
            sb.append(buildKey(ms));
        }
        return sb.toString();
    }

    private String formatSignature(MethodSignature ms) {
    String sub = ms.getSubSignature().toString();
        int spaceIdx = sub.indexOf(' ');
        String ret = spaceIdx > 0 ? sub.substring(0, spaceIdx) : "";
        String rest = spaceIdx > 0 ? sub.substring(spaceIdx + 1) : sub;
        String visibility = visibilityOf(ms);
    return ms.getDeclClassType().getFullyQualifiedName() + "#" + rest + ": " + ret + (visibility.isEmpty() ? "" : " (" + visibility + ")");
    }

    private String visibilityOf(MethodSignature ms) {
        try {
            return resolveModifier(ms);
        } catch (Exception e) {
            return "";
        }
    }

    private String resolveModifier(MethodSignature ms) {
        if (modifierCache.containsKey(ms)) return modifierCache.get(ms);
        String key = buildKey(ms);
        String mod = methodModifiers.getOrDefault(key, "");
        modifierCache.put(ms, mod);
        return mod;
    }

    private final Map<String, String> methodModifiers = new HashMap<>();
    private final Map<MethodSignature, String> modifierCache = new HashMap<>();
    private final Map<String, String> classModifiers = new HashMap<>();
    private final Map<String, List<String>> constructorsByClass = new HashMap<>();
    private final Map<String, List<String>> constructorModsByClass = new HashMap<>();
    private final Map<String, List<String>> implementationsOf = new HashMap<>();
    private final Map<String, List<String>> factoryMethodsFor = new HashMap<>();

    private String buildKey(MethodSignature ms) {
        return ms.getDeclClassType().getFullyQualifiedName() + "::" + ms.getSubSignature().toString();
    }

    private String annotateParamModifiers(String paramsCsv) {
        if (paramsCsv == null || paramsCsv.isEmpty()) return "";
        String[] parts = paramsCsv.split(",");
        List<String> out = new ArrayList<>(parts.length);
        for (String p : parts) {
            String t = p.trim();
            if (t.isEmpty()) { out.add(t); continue; }
            String base = t;
            String arraySuffix = "";
            while (base.endsWith("[]")) {
                arraySuffix += "[]";
                base = base.substring(0, base.length() - 2);
            }
            String mod = classModifiers.getOrDefault(base, "");
            if (!mod.isEmpty()) {
                out.add(base + " (" + mod + ")" + arraySuffix);
            } else {
                out.add(t);
            }
        }
        return String.join(", ", out);
    }

    private String computeProjectRootPrefix(String classFqn) {
        if (classFqn == null) return "";
        int lastDot = classFqn.lastIndexOf('.');
        String pkg = lastDot > 0 ? classFqn.substring(0, lastDot) : "";
        if (pkg.isEmpty()) return "";
        String[] segs = pkg.split("\\.");
        if (segs.length <= 2) return pkg;
        StringBuilder root = new StringBuilder();
        int take = Math.min(3, segs.length);
        for (int i = 0; i < take; i++) {
            if (i > 0) root.append('.');
            root.append(segs[i]);
        }
        return root.toString();
    }
    private boolean hasConcreteImplementation(String abstractType, Set<String> visited) {
        if (visited.contains(abstractType)) {
            return false; 
        }
        visited.add(abstractType);
        
        List<String> impls = implementationsOf.getOrDefault(abstractType, Collections.emptyList());
        for (String impl : impls) {
            String implMod = classModifiers.getOrDefault(impl, "").trim();
            if (implMod.startsWith("public") && !implMod.contains("abstract")) {
                return true; 
            }
            if (implMod.contains("abstract")) {
                if (hasConcreteImplementation(impl, visited)) {
                    return true;
                }
            }
        }
        return false;
    }
    private void collectAllImplementations(String type, Set<String> result, Set<String> visited) {
        if (visited.contains(type)) {
            return;
        }
        visited.add(type);
        List<String> impls = implementationsOf.getOrDefault(type, Collections.emptyList());
        for (String impl : impls) {
            String implMod = classModifiers.getOrDefault(impl, "").trim();
            if (implMod.startsWith("public") && !implMod.contains("abstract")) {
                result.add(impl);
            }
            collectAllImplementations(impl, result, visited);
        }
    }
}
