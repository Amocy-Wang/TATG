# Benchmark Methods

This directory documents the benchmark targets used by the artifact. The list
contains 141 method-level targets. Each row gives the project name, class name,
method name, and selector used by the evaluation scripts.

This file is provided for reviewer convenience only. The same target list is
already embedded in the runnable benchmark scripts used by the artifact.

The target list was generated from:

```bash
python scripts/run_all_hard_methods_batch.py --list-hard-methods
```

| # | Project | Class | Method | Selector |
|---:|---|---|---|---|
| 1 | `Math-2f` | `Rotation` | `getAngles` | `Rotation.getAngles` |
| 2 | `Math-2f` | `GaussNewtonOptimizer` | `doOptimize` | `GaussNewtonOptimizer.doOptimize` |
| 3 | `JacksonXml-5f` | `FromXmlParser` | `nextTextValue` | `FromXmlParser.nextTextValue` |
| 4 | `JacksonXml-5f` | `ToXmlGenerator` | `writeString` | `ToXmlGenerator.writeString` |
| 5 | `Codec-18f` | `Metaphone` | `metaphone` | `Metaphone.metaphone` |
| 6 | `Lang-4f` | `NumberUtils` | `createNumber` | `NumberUtils.createNumber` |
| 7 | `JacksonCore-26f` | `FilteringParserDelegate` | `nextToken` | `FilteringParserDelegate.nextToken` |
| 8 | `Collections-28f` | `Flat3Map` | `remove` | `Flat3Map.remove` |
| 9 | `Time-13f` | `DateTimeFormat` | `parsePatternTo` | `DateTimeFormat.parsePatternTo` |
| 10 | `Jsoup-93f` | `Entities` | `escape` | `Entities.escape` |
| 11 | `jquick-curl` | `JQuickCurlCommonVisitor` | `visitCurlCommand` | `JQuickCurlCommonVisitor.visitCurlCommand` |
| 12 | `jquick-curl` | `JQuickCurlCommonVisitor` | `visitOption` | `JQuickCurlCommonVisitor.visitOption` |
| 13 | `jquick-curl` | `JCurlCommandProcessor` | `resolveCurlCommand` | `JCurlCommandProcessor.resolveCurlCommand` |
| 14 | `adk-java` | `FunctionCallingUtils` | `buildFunctionDeclaration` | `FunctionCallingUtils.buildFunctionDeclaration` |
| 15 | `adk-java` | `FunctionCallingUtils` | `buildSchemaFromType` | `FunctionCallingUtils.buildSchemaFromType` |
| 16 | `adk-java` | `SlidingWindowEventCompactor` | `getCompactionEvents` | `SlidingWindowEventCompactor.getCompactionEvents` |
| 17 | `adk-java` | `SchemaUtils` | `validateMapOnSchema` | `SchemaUtils.validateMapOnSchema` |
| 18 | `adk-java` | `SchemaUtils` | `validateOutputSchema` | `SchemaUtils.validateOutputSchema` |
| 19 | `adk-java` | `GeminiUtil` | `sanitizeRequestForGeminiApi` | `GeminiUtil.sanitizeRequestForGeminiApi` |
| 20 | `adk-java` | `GeminiUtil` | `stripThoughts` | `GeminiUtil.stripThoughts` |
| 21 | `adk-java` | `ChatCompletionsRequest` | `fromLlmRequest` | `ChatCompletionsRequest.fromLlmRequest` |
| 22 | `adk-java` | `Contents` | `rearrangeEventsForLatestFunctionResponse` | `Contents.rearrangeEventsForLatestFunctionResponse` |
| 23 | `adk-java` | `BaseAgent` | `findAgent` | `BaseAgent.findAgent` |
| 24 | `batch-processing-gateway` | `SubmissionSummary` | `copyFrom` | `SubmissionSummary.copyFrom` |
| 25 | `batch-processing-gateway` | `ApplicationSubmissionHelper` | `getDriverSpec` | `ApplicationSubmissionHelper.getDriverSpec` |
| 26 | `batch-processing-gateway` | `ApplicationSubmissionHelper` | `getExecutorSpec` | `ApplicationSubmissionHelper.getExecutorSpec` |
| 27 | `batch-processing-gateway` | `ApplicationSubmissionHelper` | `getSparkConf` | `ApplicationSubmissionHelper.getSparkConf` |
| 28 | `batch-processing-gateway` | `ApplicationSubmissionHelper` | `populateEnv` | `ApplicationSubmissionHelper.populateEnv` |
| 29 | `batch-processing-gateway` | `SparkPodSpec` | `copyFrom` | `SparkPodSpec.copyFrom` |
| 30 | `commons-cli` | `HelpFormatter` | `renderOptions` | `HelpFormatter.renderOptions` |
| 31 | `commons-cli` | `PatternOptionBuilder` | `getValueType` | `PatternOptionBuilder.getValueType` |
| 32 | `commons-codec` | `Base32` | `decode` | `Base32.decode` |
| 33 | `commons-codec` | `Base32` | `encode` | `Base32.encode` |
| 34 | `commons-codec` | `Base64` | `decode` | `Base64.decode` |
| 35 | `commons-codec` | `Base64` | `encode` | `Base64.encode` |
| 36 | `commons-codec` | `Md5Crypt` | `md5Crypt` | `Md5Crypt.md5Crypt` |
| 37 | `commons-codec` | `MurmurHash3` | `hash128x64Internal` | `MurmurHash3.hash128x64Internal` |
| 38 | `commons-codec` | `Sha2Crypt` | `sha2Crypt` | `Sha2Crypt.sha2Crypt` |
| 39 | `commons-codec` | `ColognePhonetic` | `colognePhonetic` | `ColognePhonetic.colognePhonetic` |
| 40 | `commons-codec` | `DaitchMokotoffSoundex` | `soundex` | `DaitchMokotoffSoundex.soundex` |
| 41 | `commons-codec` | `DoubleMetaphone` | `doubleMetaphone` | `DoubleMetaphone.doubleMetaphone` |
| 42 | `commons-codec` | `DoubleMetaphone` | `handleC` | `DoubleMetaphone.handleC` |
| 43 | `commons-codec` | `DoubleMetaphone` | `handleG` | `DoubleMetaphone.handleG` |
| 44 | `commons-codec` | `DoubleMetaphone` | `handleS` | `DoubleMetaphone.handleS` |
| 45 | `commons-codec` | `Nysiis` | `nysiis` | `Nysiis.nysiis` |
| 46 | `commons-codec` | `Nysiis` | `transcodeRemaining` | `Nysiis.transcodeRemaining` |
| 47 | `commons-codec` | `Rule` | `pattern` | `Rule.pattern` |
| 48 | `commons-codec` | `QuotedPrintableCodec` | `encodeQuotedPrintable` | `QuotedPrintableCodec.encodeQuotedPrintable` |
| 49 | `commons-collections` | `IteratorUtils` | `getIterator` | `IteratorUtils.getIterator` |
| 50 | `commons-collections` | `TreeBidiMap` | `doRedBlackDelete` | `TreeBidiMap.doRedBlackDelete` |
| 51 | `commons-collections` | `TreeBidiMap` | `swapPosition` | `TreeBidiMap.swapPosition` |
| 52 | `commons-collections` | `Flat3MapNew` | `containsKey` | `Flat3MapNew.containsKey` |
| 53 | `commons-collections` | `Flat3MapNew` | `containsValue` | `Flat3MapNew.containsValue` |
| 54 | `commons-collections` | `Flat3MapNew` | `equals` | `Flat3MapNew.equals` |
| 55 | `commons-collections` | `Flat3MapNew` | `get` | `Flat3MapNew.get` |
| 56 | `commons-collections` | `Flat3MapNew` | `put` | `Flat3MapNew.put` |
| 57 | `commons-collections` | `Flat3MapNew` | `remove` | `Flat3MapNew.remove` |
| 58 | `commons-collections` | `SequencesComparator` | `getMiddleSnake` | `SequencesComparator.getMiddleSnake` |
| 59 | `commons-collections` | `AbstractPatriciaTrie` | `higherEntry` | `AbstractPatriciaTrie.higherEntry` |
| 60 | `commons-collections` | `AbstractPatriciaTrie` | `nextEntryImpl` | `AbstractPatriciaTrie.nextEntryImpl` |
| 61 | `commons-collections` | `AbstractPatriciaTrie` | `put` | `AbstractPatriciaTrie.put` |
| 62 | `commons-collections` | `AbstractPatriciaTrie` | `removeInternalEntry` | `AbstractPatriciaTrie.removeInternalEntry` |
| 63 | `commons-csv` | `CSVFormat` | `printWithQuotes` | `CSVFormat.printWithQuotes` |
| 64 | `commons-csv` | `CSVFormat` | `toString` | `CSVFormat.toString` |
| 65 | `commons-csv` | `CSVParser` | `createHeaders` | `CSVParser.createHeaders` |
| 66 | `commons-csv` | `CSVParser` | `nextRecord` | `CSVParser.nextRecord` |
| 67 | `commons-csv` | `Lexer` | `nextToken` | `Lexer.nextToken` |
| 68 | `commons-csv` | `Lexer` | `parseEncapsulatedToken` | `Lexer.parseEncapsulatedToken` |
| 69 | `datafaker` | `FakeValues` | `toJavaNames` | `FakeValues.toJavaNames` |
| 70 | `datafaker` | `FakeValuesService` | `bothify` | `FakeValuesService.bothify` |
| 71 | `datafaker` | `FakeValuesService` | `javaNameToYamlName` | `FakeValuesService.javaNameToYamlName` |
| 72 | `datafaker` | `FakeValuesService` | `resolveExpression` | `FakeValuesService.resolveExpression` |
| 73 | `datafaker` | `SqlTransformer` | `handlePrimitivesInArray` | `SqlTransformer.handlePrimitivesInArray` |
| 74 | `gson/gson` | `$Gson$Types` | `equals` | `$Gson$Types.equals` |
| 75 | `gson/gson` | `$Gson$Types` | `getGenericSupertype` | `$Gson$Types.getGenericSupertype` |
| 76 | `gson/gson` | `$Gson$Types` | `resolve` | `$Gson$Types.resolve` |
| 77 | `gson/gson` | `ConstructorConstructor` | `newDefaultImplementationConstructor` | `ConstructorConstructor.newDefaultImplementationConstructor` |
| 78 | `gson/gson` | `Excluder` | `excludeField` | `Excluder.excludeField` |
| 79 | `gson/gson` | `LinkedTreeMap` | `find` | `LinkedTreeMap.find` |
| 80 | `gson/gson` | `LinkedTreeMap` | `rebalance` | `LinkedTreeMap.rebalance` |
| 81 | `gson/gson` | `Primitives` | `unwrap` | `Primitives.unwrap` |
| 82 | `gson/gson` | `Primitives` | `wrap` | `Primitives.wrap` |
| 83 | `gson/gson` | `JsonTreeReader` | `peek` | `JsonTreeReader.peek` |
| 84 | `gson/gson` | `ObjectTypeAdapter` | `read` | `ObjectTypeAdapter.read` |
| 85 | `gson/gson` | `ReflectiveTypeAdapterFactory` | `getBoundFields` | `ReflectiveTypeAdapterFactory.getBoundFields` |
| 86 | `gson/gson` | `TypeToken` | `isAssignableFrom` | `TypeToken.isAssignableFrom` |
| 87 | `gson/gson` | `JsonReader` | `doPeek` | `JsonReader.doPeek` |
| 88 | `gson/gson` | `JsonReader` | `nextNonWhitespace` | `JsonReader.nextNonWhitespace` |
| 89 | `gson/gson` | `JsonReader` | `nextUnquotedValue` | `JsonReader.nextUnquotedValue` |
| 90 | `gson/gson` | `JsonReader` | `peek` | `JsonReader.peek` |
| 91 | `gson/gson` | `JsonReader` | `peekNumber` | `JsonReader.peekNumber` |
| 92 | `gson/gson` | `JsonReader` | `readEscapeCharacter` | `JsonReader.readEscapeCharacter` |
| 93 | `gson/gson` | `JsonReader` | `skipValue` | `JsonReader.skipValue` |
| 94 | `jdom2` | `Element` | `getNamespace` | `Element.getNamespace` |
| 95 | `jdom2` | `Element` | `getNamespacesInScope` | `Element.getNamespacesInScope` |
| 96 | `jdom2` | `StringBin` | `locate` | `StringBin.locate` |
| 97 | `jdom2` | `Verifier` | `isXMLExtender` | `Verifier.isXMLExtender` |
| 98 | `jdom2` | `Verifier` | `isXMLPublicIDCharacter` | `Verifier.isXMLPublicIDCharacter` |
| 99 | `jdom2` | `StAXStreamBuilder` | `processPrunableElement` | `StAXStreamBuilder.processPrunableElement` |
| 100 | `jdom2` | `SAXHandler` | `startElement` | `SAXHandler.startElement` |
| 101 | `jdom2` | `DTDParser` | `formatInternal` | `DTDParser.formatInternal` |
| 102 | `jdom2` | `Format` | `escapeAttribute` | `Format.escapeAttribute` |
| 103 | `jdom2` | `Format` | `escapeText` | `Format.escapeText` |
| 104 | `jdom2` | `AbstractFormattedWalker` | `next` | `AbstractFormattedWalker.next` |
| 105 | `jdom2` | `AbstractSAXOutputProcessor` | `printContent` | `AbstractSAXOutputProcessor.printContent` |
| 106 | `jdom2` | `AbstractStAXEventProcessor` | `printContent` | `AbstractStAXEventProcessor.printContent` |
| 107 | `jdom2` | `AbstractStAXEventProcessor` | `printDocument` | `AbstractStAXEventProcessor.printDocument` |
| 108 | `jdom2` | `AbstractStAXStreamProcessor` | `printContent` | `AbstractStAXStreamProcessor.printContent` |
| 109 | `jdom2` | `AbstractStAXStreamProcessor` | `printDocument` | `AbstractStAXStreamProcessor.printDocument` |
| 110 | `jdom2` | `AbstractXMLOutputProcessor` | `printContent` | `AbstractXMLOutputProcessor.printContent` |
| 111 | `jdom2` | `AbstractXMLOutputProcessor` | `printDocument` | `AbstractXMLOutputProcessor.printDocument` |
| 112 | `jdom2` | `WalkerNORMALIZE` | `analyzeMultiText` | `WalkerNORMALIZE.analyzeMultiText` |
| 113 | `jdom2` | `WalkerTRIM` | `analyzeMultiText` | `WalkerTRIM.analyzeMultiText` |
| 114 | `jdom2` | `XPathHelper` | `getSingleStep` | `XPathHelper.getSingleStep` |
| 115 | `ruler` | `ByteMachine` | `addEndOfMatch` | `ByteMachine.addEndOfMatch` |
| 116 | `ruler` | `ByteMachine` | `addMatchReferences` | `ByteMachine.addMatchReferences` |
| 117 | `ruler` | `ByteMachine` | `addRangePattern` | `ByteMachine.addRangePattern` |
| 118 | `ruler` | `ByteMachine` | `deleteRangePattern` | `ByteMachine.deleteRangePattern` |
| 119 | `ruler` | `ByteMachine` | `doTransitionOn` | `ByteMachine.doTransitionOn` |
| 120 | `ruler` | `ByteMachine` | `findMatchPattern` | `ByteMachine.findMatchPattern` |
| 121 | `ruler` | `ByteMachine` | `findRangePattern` | `ByteMachine.findRangePattern` |
| 122 | `ruler` | `ByteMachine` | `updateMatchReferences` | `ByteMachine.updateMatchReferences` |
| 123 | `ruler` | `GenericMachine` | `deleteStep` | `GenericMachine.deleteStep` |
| 124 | `ruler` | `JsonRuleCompiler` | `processMatchExpression` | `JsonRuleCompiler.processMatchExpression` |
| 125 | `ruler` | `JsonRuleCompiler` | `writeRules` | `JsonRuleCompiler.writeRules` |
| 126 | `ruler` | `MachineComplexityEvaluator` | `evaluate` | `MachineComplexityEvaluator.evaluate` |
| 127 | `ruler` | `MachineComplexityEvaluator` | `getMatchesAccessibleFromEachTransition` | `MachineComplexityEvaluator.getMatchesAccessibleFromEachTransition` |
| 128 | `ruler` | `RuleCompiler` | `processMatchExpression` | `RuleCompiler.processMatchExpression` |
| 129 | `ruler` | `RuleCompiler` | `writeRules` | `RuleCompiler.writeRules` |
| 130 | `ruler` | `Ruler` | `matches` | `Ruler.matches` |
| 131 | `windward` | `AbstractRouterGroup` | `matchRouter` | `AbstractRouterGroup.matchRouter` |
| 132 | `windward` | `AntPathMatcher` | `isMatch` | `AntPathMatcher.isMatch` |
| 133 | `agentscope-java` | `ReActAgent` | `doCall` | `ReActAgent.doCall` |
| 134 | `agentscope-java` | `OpenAIResponseParser` | `parseCompletionResponse` | `OpenAIResponseParser.parseCompletionResponse` |
| 135 | `agentscope-java` | `OpenAIResponseParser` | `parseChunkResponse` | `OpenAIResponseParser.parseChunkResponse` |
| 136 | `agentscope-java` | `PlanNotebook` | `updateSubtaskState` | `PlanNotebook.updateSubtaskState` |
| 137 | `agentscope-java` | `PlanNotebook` | `reviseCurrentPlan` | `PlanNotebook.reviseCurrentPlan` |
| 138 | `a2a-java/http-client` | `ServerSentEventParser` | `processLine` | `ServerSentEventParser.processLine` |
| 139 | `a2a-java/http-client` | `ServerSentEventParser` | `dispatchEvent` | `ServerSentEventParser.dispatchEvent` |
| 140 | `a2a-java/server-common` | `AgentCardValidator` | `validateTransportConfiguration` | `AgentCardValidator.validateTransportConfiguration` |
| 141 | `a2a-java/server-common` | `DefaultRequestHandler` | `onMessageSend` | `DefaultRequestHandler.onMessageSend` |
