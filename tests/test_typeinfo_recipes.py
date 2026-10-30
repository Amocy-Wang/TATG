from pathlib import Path

from tatg.analysis.typeinfo import build_type_aware_context
from tatg.analysis.typeinfo.models import ConstructorInfo, ExecutionRecipe, TypeShape
from tatg.analysis.typeinfo.recipe_synthesizer import (
    _legacy_name_heuristics_enabled,
    _scan_project_implementation_hints,
    _target_dependency_context,
    infer_capsule_required_call_order,
    render_execution_capsules_section,
    render_execution_recipes_section,
    render_target_receipt_contract,
)


def test_recipe_synthesizer_has_no_benchmark_specific_class_names():
    source = Path(__file__).parents[1] / "src/tatg/analysis/typeinfo/recipe_synthesizer.py"
    text = source.read_text(encoding="utf-8")

    forbidden_names = [
        "FromXmlParser",
        "ToXmlGenerator",
        "GaussNewtonOptimizer",
        "Rotation",
        "JacksonXml",
        "Math-2f",
    ]
    for name in forbidden_names:
        assert name not in text


def test_recipe_legacy_name_heuristics_cannot_be_reenabled_by_environment(monkeypatch):
    monkeypatch.setenv("TATG_ENABLE_LEGACY_NAME_HEURISTICS", "1")

    assert not _legacy_name_heuristics_enabled()


def test_dependency_context_does_not_infer_protocol_from_method_names(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/NamedLikeAProtocol.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class NamedLikeAProtocol {
    public NamedLikeAProtocol() {}
    public void start() {}
    public int next() { return 1; }
    public void close() {}
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="next")
    dependency_context = _target_dependency_context(
        ctx.type_shapes["NamedLikeAProtocol"]
    )

    assert any("construction" in item for item in dependency_context)
    assert not any("lifecycle/protocol" in item for item in dependency_context)
    assert not any("start" in item or "next" in item or "close" in item for item in dependency_context)


def test_execution_recipes_include_inherited_setup_and_dimensions(tmp_path: Path):
    base = tmp_path / "src/main/java/com/acme/BaseSolver.java"
    child = tmp_path / "src/main/java/com/acme/ConcreteSolver.java"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        """
package com.acme;

public abstract class BaseSolver<T> {
    protected double[] target;
    protected double[] start;

    public T withTarget(double[] target) {
        this.target = target;
        return (T) this;
    }

    public T withStart(double[] start) {
        this.start = start;
        return (T) this;
    }
}
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
package com.acme;

public class ConcreteSolver extends BaseSolver<ConcreteSolver> {
    public int doRun(double[] weights) {
        if (target == null) {
            throw new NullPointerException();
        }
        if (weights.length != target.length) {
            throw new IllegalArgumentException();
        }
        return start.length + target.length;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(child, tmp_path, method_name="doRun")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "BaseSolver.withTarget" in rendered
    assert "BaseSolver.withStart" in rendered
    assert "shape/dimension compatibility" in rendered
    assert "non-null" in rendered


def test_execution_recipes_do_not_treat_string_length_boundary_as_shape(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/Numbers.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class Numbers {
    public Number createNumber(String str, int expPos) {
        if (str == null) {
            return null;
        }
        if (expPos > str.length()) {
            throw new NumberFormatException(str);
        }
        if (str.startsWith("0x")) {
            return Long.valueOf(str.substring(2), 16);
        }
        return Integer.valueOf(str);
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="createNumber")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "shape/dimension compatibility" not in rendered
    assert "guarded value absent or null" in rendered


def test_execution_recipes_include_compute_callback_and_resource_guidance(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    base = tmp_path / "src/main/java/com/acme/BaseOptimizer.java"
    child = tmp_path / "src/main/java/com/acme/StreamingOptimizer.java"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        """
package com.acme;

public abstract class BaseOptimizer<T> {
    public T withModelAndJacobian(ModelFunction model, JacobianFunction jacobian) {
        return (T) this;
    }

    public T withConvergenceChecker(ConvergenceChecker checker) {
        return (T) this;
    }
}

interface ModelFunction {}
interface JacobianFunction {}
interface ConvergenceChecker {}
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
package com.acme;

import javax.xml.stream.XMLStreamWriter;

public class StreamingOptimizer extends BaseOptimizer<StreamingOptimizer> {
    public StreamingOptimizer(FrameworkContext context, XMLStreamWriter writer) {}

    public int doOptimize(double[] target) {
        if (target == null) {
            throw new NullPointerException();
        }
        return target.length;
    }
}

class FrameworkContext {}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(child, tmp_path, method_name="doOptimize")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "function/callback implementations" in rendered
    assert "convergence/checker" in rendered
    assert "StringWriter" in rendered
    assert "owner factory" in rendered


def test_abstract_construction_recipe_names_known_concrete_subclass(tmp_path: Path):
    base = tmp_path / "src/main/java/com/acme/BaseAgent.java"
    child = tmp_path / "src/main/java/com/acme/SequentialAgent.java"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        """
package com.acme;

public abstract class BaseAgent {
    public BaseAgent(String name) {}
    public abstract BaseAgent findAgent(String name);
}
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
package com.acme;

public class SequentialAgent extends BaseAgent {
    public SequentialAgent() { super("seq"); }
    public BaseAgent findAgent(String name) { return this; }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(base, tmp_path, method_name="findAgent")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "known concrete subclass candidates" in rendered
    assert "SequentialAgent" in rendered


def test_constructor_dependency_does_not_advertise_recursive_abstract_wrapper(
    tmp_path: Path,
):
    base = tmp_path / "src/main/java/com/acme/Protocol.java"
    wrapper = tmp_path / "src/main/java/com/acme/ProtocolDelegate.java"
    owner = tmp_path / "src/main/java/com/acme/Owner.java"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        """
package com.acme;

public abstract class Protocol {
    public abstract String next();
}
""".strip(),
        encoding="utf-8",
    )
    wrapper.write_text(
        """
package com.acme;

public class ProtocolDelegate extends Protocol {
    private final Protocol delegate;
    public ProtocolDelegate(Protocol delegate) { this.delegate = delegate; }
    public String next() { return delegate.next(); }
}
""".strip(),
        encoding="utf-8",
    )
    owner.write_text(
        """
package com.acme;

public class Owner {
    public Owner(Protocol protocol) {}
    public String target() { return "ok"; }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(owner, tmp_path, method_name="target")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "prefer a source-visible concrete implementation for the valid baseline: ProtocolDelegate" not in rendered


def test_project_subtype_scan_reuses_independent_construction_gate(tmp_path: Path):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "ProtocolDelegate.java").write_text(
        """
package com.acme;
public class ProtocolDelegate extends Protocol {
    public ProtocolDelegate(Protocol delegate) {}
}
""".strip(),
        encoding="utf-8",
    )
    (source_root / "ConcreteProtocol.java").write_text(
        """
package com.acme;
public class ConcreteProtocol extends Protocol {
    public ConcreteProtocol() {}
}
""".strip(),
        encoding="utf-8",
    )
    protocol = TypeShape(
        class_name="Protocol",
        fqn="com.acme.Protocol",
        package="com.acme",
        is_abstract=True,
    )
    delegate = TypeShape(
        class_name="ProtocolDelegate",
        fqn="com.acme.ProtocolDelegate",
        package="com.acme",
        superclass="Protocol",
        constructors=[
            ConstructorInfo(params=[("Protocol", "delegate")], visibility="public")
        ],
    )
    concrete = TypeShape(
        class_name="ConcreteProtocol",
        fqn="com.acme.ConcreteProtocol",
        package="com.acme",
        superclass="Protocol",
        constructors=[ConstructorInfo(params=[], visibility="public")],
    )

    hints = _scan_project_implementation_hints(
        protocol,
        tmp_path,
        {
            "ProtocolDelegate": delegate,
            "ConcreteProtocol": concrete,
        },
    )

    assert not any("ProtocolDelegate" in hint for hint in hints)
    assert any("ConcreteProtocol" in hint for hint in hints)


def test_numeric_receipt_includes_callback_method_signatures(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    base = tmp_path / "src/main/java/com/acme/BaseOptimizer.java"
    child = tmp_path / "src/main/java/com/acme/ConcreteOptimizer.java"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        """
package com.acme;

public abstract class BaseOptimizer<T> {
    public T withModelAndJacobian(ModelFunction model, JacobianFunction jacobian) {
        return (T) this;
    }
}

""".strip(),
        encoding="utf-8",
    )
    (base.parent / "ModelFunction.java").write_text(
        """
package com.acme;

public interface ModelFunction {
    double[] value(double[] point);
}
""".strip(),
        encoding="utf-8",
    )
    (base.parent / "JacobianFunction.java").write_text(
        """
package com.acme;

public interface JacobianFunction {
    double[][] value(double[] point);
}
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
package com.acme;

public class ConcreteOptimizer extends BaseOptimizer<ConcreteOptimizer> {
    public double doOptimize(double[] start) {
        if (start.length == 0) {
            throw new IllegalArgumentException();
        }
        return start[0];
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(child, tmp_path, method_name="doOptimize")
    rendered = render_target_receipt_contract(ctx.execution_recipes, method_name="doOptimize")

    assert "callback-contracts" in rendered
    assert "ModelFunction.value(double[] point) -> double[]" in rendered
    assert "JacobianFunction.value(double[] point) -> double[][]" in rendered


def test_execution_recipes_include_constant_path_knobs(tmp_path: Path):
    mode = tmp_path / "src/main/java/com/acme/Mode.java"
    target = tmp_path / "src/main/java/com/acme/ModeInterpreter.java"
    mode.parent.mkdir(parents=True, exist_ok=True)
    mode.write_text(
        """
package com.acme;

public class Mode {
    public static final Mode FAST = new Mode();
    public static final Mode SAFE = new Mode();
}
""".strip(),
        encoding="utf-8",
    )
    target.write_text(
        """
package com.acme;

public class ModeInterpreter {
    public int score(Mode mode, boolean strict) {
        if (strict) {
            return 1;
        }
        if (mode == null) {
            throw new IllegalArgumentException();
        }
        return 2;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(target, tmp_path, method_name="score")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "FAST" in rendered
    assert "SAFE" in rendered
    assert "both true and false" in rendered


def test_execution_recipes_include_explicit_enum_branch_family(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/OrderInterpreter.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

enum Order { XYZ, XZY, YXZ, YZX, ZXY, ZYX }

public class OrderInterpreter {
    public int score(Order order) {
        if (order == Order.XYZ) {
            return 1;
        } else if (order == Order.XZY) {
            return 2;
        } else if (order == Order.YXZ) {
            return 3;
        } else if (order == Order.YZX) {
            return 4;
        } else if (order == Order.ZXY) {
            return 5;
        } else {
            return 6;
        }
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="score")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "explicit branch family" in rendered
    assert "Order.XYZ" in rendered
    assert "Order.ZXY" in rendered


def test_protocol_like_names_do_not_create_state_sequence_recipe(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/TokenReader.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class TokenReader {
    private int currentToken;
    private boolean closed;

    public void open(String input) {
        currentToken = 0;
    }

    public int nextToken() {
        currentToken++;
        return currentToken;
    }

    public String getText() {
        if (closed) {
            throw new IllegalStateException();
        }
        return currentToken > 0 ? "text" : "";
    }

    public void close() {
        closed = true;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="getText")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "state machine" not in rendered
    assert "currentToken" in rendered
    assert "[category: state-sequence]" not in rendered


def test_name_driven_state_and_token_recipes_are_disabled_by_default(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/TokenReader.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;
public class TokenReader {
    private int currentToken;
    public int nextToken() {
        currentToken++;
        switch (currentToken) {
            case 1: return 1;
            case 2: return 2;
            default: return 0;
        }
    }
    public String getText() { return currentToken > 0 ? "text" : ""; }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="getText")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "[category: state-sequence]" not in rendered
    assert "[category: token-parser-matrix]" not in rendered


def test_specialized_query_name_does_not_invent_advance_precondition(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/TokenReader.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class TokenReader {
    private int currentToken;
    private String currentText;

    public int nextToken() {
        currentToken++;
        return currentToken;
    }

    public String nextTextValue() {
        if (currentToken == 0) {
            return null;
        }
        return currentText;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="nextTextValue")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "first call nextToken()" not in rendered
    assert "primed-state value behavior" not in rendered
    assert "currentToken" in rendered


def test_execution_recipes_preserve_overload_path_axes_without_name_state_recipe(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/StatefulCodec.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class StatefulCodec {
    private int currentToken;

    public void nextToken() {
        currentToken++;
    }

    public String readValue(boolean strict) {
        if (strict) {
            return "strict";
        }
        return currentToken > 0 ? "value" : "";
    }

    public String readValue(String fallback) {
        if (fallback == null) {
            throw new IllegalArgumentException();
        }
        return fallback;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="readValue")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "currentToken" in rendered
    assert "nextToken" not in rendered
    assert "boolean parameter strict" in rendered
    assert "guarded value absent or null" in rendered


def test_execution_recipes_emit_generic_numeric_solver_receipt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    source = tmp_path / "src/main/java/com/acme/RankedStepper.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class RankedStepper {
    private double[] target;
    private double[] start;
    private Matrix weight;
    private VectorFunction model;
    private MatrixFunction jacobian;
    private ConvergenceChecker checker;

    public RankedStepper withTarget(double[] values) { this.target = values; return this; }
    public RankedStepper withStart(double[] values) { this.start = values; return this; }
    public RankedStepper withWeight(Matrix values) { this.weight = values; return this; }
    public RankedStepper withModel(VectorFunction fn) { this.model = fn; return this; }
    public RankedStepper withJacobian(MatrixFunction fn) { this.jacobian = fn; return this; }
    public RankedStepper withChecker(ConvergenceChecker fn) { this.checker = fn; return this; }

    public double[] computeStep() {
        if (checker == null) {
            throw new IllegalStateException();
        }
        if (weight.getRowDimension() != target.length) {
            throw new IllegalArgumentException();
        }
        double[] objective = model.value(start);
        double[][] j = jacobian.value(start);
        Matrix normal = weight.multiply(j).solve(objective);
        for (int iteration = 0; !checker.converged(iteration, start, objective); iteration++) {
            return normal.toArray();
        }
        return objective;
    }
}

interface Matrix { int getRowDimension(); Matrix multiply(double[][] data); Matrix solve(double[] data); double[] toArray(); }
interface VectorFunction { double[] value(double[] point); }
interface MatrixFunction { double[][] value(double[] point); }
interface ConvergenceChecker { boolean converged(int iteration, double[] previous, double[] current); }
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="computeStep")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "numeric-shape-solver" in rendered or "numeric_shape_solver" in rendered
    assert "full-column-rank" in rendered
    assert "observations" in rendered
    assert "parameters" in rendered
    assert "ConvergenceChecker" not in rendered or "checker" in rendered


def test_protocol_like_source_does_not_emit_name_inferred_receipt(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/EventCursor.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class EventCursor {
    private int currentIndex;
    private String currentValue;
    private boolean closed;

    public void open(String input) {
        currentIndex = 0;
    }

    public int read() {
        currentIndex++;
        currentValue = "value";
        return currentIndex;
    }

    public String readEntry() {
        if (closed) {
            throw new IllegalStateException();
        }
        if (currentIndex == 0) {
            return null;
        }
        return currentValue;
    }

    public void close() {
        closed = true;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="readEntry")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "stateful-cursor-or-protocol" not in rendered
    assert "stateful_cursor_or_protocol" not in rendered
    assert "call generic advance method read" not in rendered
    assert "readEntry" in rendered


def test_target_receipt_contract_does_not_invent_name_based_call_order(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/EventCursor.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class EventCursor {
    private int currentIndex;
    private String currentValue;

    public int read() { currentIndex++; currentValue = "value"; return currentIndex; }

    public String readEntry() {
        if (currentIndex == 0) {
            return null;
        }
        return currentValue;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="readEntry")
    rendered = render_target_receipt_contract(ctx.execution_recipes, method_name="readEntry")

    assert "[Target Execution Receipt]" in rendered
    assert "readEntry" in rendered
    assert "A mention in the test name" in rendered
    assert "call generic advance method read" not in rendered


def test_method_names_do_not_create_stateful_capsule_or_hard_call_order(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/EventCursor.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class EventCursor {
    private int currentIndex;
    private String currentValue;

    public EventCursor(String input) {}
    public int read() { currentIndex++; currentValue = "value"; return currentIndex; }

    public String readEntry() {
        if (currentIndex == 0) {
            return null;
        }
        return currentValue;
    }

    public void close() {}
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="readEntry")
    rendered = render_execution_capsules_section(
        ctx.execution_recipes,
        method_name="readEntry",
        target_shape=ctx.type_shapes["EventCursor"],
    )

    assert "stateful_resource_protocol" not in rendered
    assert infer_capsule_required_call_order(
        ctx.execution_recipes,
        method_name="readEntry",
    ) == []


def test_source_verified_receipt_can_create_hard_call_order():
    recipe = ExecutionRecipe(
        method_name="inspect",
        category="source-state-sequence",
        title="source-proven transition",
        receipt={
            "source_verified_required_order": [
                ["advanceState", "inspect"],
            ],
        },
    )

    assert infer_capsule_required_call_order(
        [recipe],
        method_name="inspect",
    ) == [("advanceState", "inspect")]


def test_output_value_writer_names_do_not_create_protocol_setup(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/EventGenerator.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class EventGenerator {
    private String nextName;
    private int writeContext;

    public EventGenerator(Object sink) {}

    public void setNextName(String name) {
        nextName = name;
    }

    public void writeStartObject() {
        if (nextName == null) {
            throw new IllegalStateException("missing name");
        }
        writeContext++;
    }

    public void writeString(String value) {
        if (nextName == null) {
            throw new IllegalStateException("missing name");
        }
        writeContext++;
    }

    public void flush() {}
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="writeString")
    receipt = render_target_receipt_contract(ctx.execution_recipes, method_name="writeString")
    capsule = render_execution_capsules_section(
        ctx.execution_recipes,
        method_name="writeString",
        target_shape=ctx.type_shapes["EventGenerator"],
    )

    assert "before calling value-writing target writeString" not in receipt
    assert "do not call structural start methods before writeString" not in receipt
    assert "before calling value-writing target writeString" not in capsule
    assert "call the capsule advance method before the target call" not in capsule


def test_multiple_protocol_like_names_do_not_create_hard_call_order(
    tmp_path: Path,
):
    cases = [
        (
            "StreamCursor",
            "readEntry",
            "read",
            """
package com.acme;

public class StreamCursor {
    private int currentIndex;
    private String currentValue;
    private boolean closed;

    public void open(String input) { currentIndex = 0; }
    public int read() { currentIndex++; currentValue = "value"; return currentIndex; }
    public String readEntry() {
        if (closed) { throw new IllegalStateException(); }
        return currentIndex > 0 ? currentValue : null;
    }
    public void close() { closed = true; }
}
""".strip(),
        ),
        (
            "FrameWriter",
            "writeValue",
            "writeStart",
            """
package com.acme;

public class FrameWriter {
    private int depth;
    private boolean closed;
    private String buffer;

    public void open(StringBuilder sink) { buffer = ""; }
    public void writeStart() { depth++; }
    public String writeValue(String value) {
        if (closed) { throw new IllegalStateException(); }
        if (depth == 0) { return "missing-frame"; }
        buffer = value;
        return buffer;
    }
    public void finish() { closed = true; }
}
""".strip(),
        ),
        (
            "PacketConsumer",
            "peekItem",
            "consume",
            """
package com.acme;

public class PacketConsumer {
    private int pos;
    private String currentItem;
    private boolean finished;

    public void begin(byte[] input) { pos = 0; }
    public void consume() { pos++; currentItem = "packet"; }
    public String peekItem() {
        if (finished) { throw new IllegalStateException(); }
        return pos > 0 ? currentItem : null;
    }
    public void end() { finished = true; }
}
""".strip(),
        ),
    ]

    for class_name, method_name, advance_method, source_text in cases:
        source = tmp_path / f"src/main/java/com/acme/{class_name}.java"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source_text, encoding="utf-8")

        ctx = build_type_aware_context(source, tmp_path, method_name=method_name)
        rendered = render_execution_capsules_section(
            ctx.execution_recipes,
            method_name=method_name,
            target_shape=ctx.type_shapes[class_name],
        )

        assert "stateful_resource_protocol" not in rendered
        assert infer_capsule_required_call_order(
            ctx.execution_recipes,
            method_name=method_name,
        ) == []


def test_capsule_required_call_order_does_not_gate_plain_get_accessors(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/MapLike.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class MapLike {
    private int nextIndex;
    private String key;

    public void reset() { nextIndex = 0; }
    public Object next() { nextIndex++; return key; }
    public Object get(Object key) { return "value"; }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="get")

    assert infer_capsule_required_call_order(ctx.execution_recipes, method_name="get") == []


def test_execution_capsule_routes_numeric_targets_without_class_names(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    base = tmp_path / "src/main/java/com/acme/BaseNumeric.java"
    child = tmp_path / "src/main/java/com/acme/RankSolver.java"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        """
package com.acme;

public abstract class BaseNumeric<T> {
    public T withTarget(double[] target) { return (T) this; }
    public T withStart(double[] start) { return (T) this; }
    public T withModel(Model model) { return (T) this; }
    public T withJacobian(Jacobian jacobian) { return (T) this; }
    public T withConvergenceChecker(Checker checker) { return (T) this; }
}

interface Model {}
interface Jacobian {}
interface Checker {}
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
package com.acme;

public class RankSolver extends BaseNumeric<RankSolver> {
    public double[] doOptimize(double[] target, double[][] jacobian, double[] start) {
        if (target.length != jacobian.length) { throw new IllegalArgumentException(); }
        if (start.length != jacobian[0].length) { throw new IllegalArgumentException(); }
        while (start.length > 0) { return target; }
        return start;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(child, tmp_path, method_name="doOptimize")
    rendered = render_execution_capsules_section(
        ctx.execution_recipes,
        method_name="doOptimize",
        target_shape=ctx.type_shapes["RankSolver"],
    )

    assert "[Execution Capsule Skeleton]" in rendered
    assert "numeric_rank_safe_fixture" in rendered
    assert "dimension" in rendered
    assert "rank" in rendered
    assert "compiled-test-template" in rendered
    assert "doOptimize" in rendered
    assert "GaussNewtonOptimizer" not in rendered
    assert infer_capsule_required_call_order(ctx.execution_recipes, method_name="doOptimize") == []


def test_execution_capsule_routes_numeric_solver_families_without_benchmark_names(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    cases = [
        (
            "ResidualUpdater",
            "computeUpdate",
            """
package com.acme;

public class ResidualUpdater {
    private double[] target;
    private double[] start;
    private Matrix weight;
    private VectorFunction model;
    private MatrixFunction jacobian;
    private Checker checker;

    public ResidualUpdater withTarget(double[] target) { this.target = target; return this; }
    public ResidualUpdater withStart(double[] start) { this.start = start; return this; }
    public ResidualUpdater withWeight(Matrix weight) { this.weight = weight; return this; }
    public ResidualUpdater withModel(VectorFunction model) { this.model = model; return this; }
    public ResidualUpdater withJacobian(MatrixFunction jacobian) { this.jacobian = jacobian; return this; }
    public ResidualUpdater withChecker(Checker checker) { this.checker = checker; return this; }

    public double[] computeUpdate() {
        if (target.length != weight.getRowDimension()) { throw new IllegalArgumentException(); }
        double[] residual = model.value(start);
        double[][] j = jacobian.value(start);
        if (!checker.converged(0, start, residual)) { return weight.solve(j, residual); }
        return residual;
    }
}

interface Matrix { int getRowDimension(); double[] solve(double[][] jacobian, double[] residual); }
interface VectorFunction { double[] value(double[] point); }
interface MatrixFunction { double[][] value(double[] point); }
interface Checker { boolean converged(int iteration, double[] previous, double[] current); }
""".strip(),
        ),
        (
            "NormalEquationSolver",
            "solveStep",
            """
package com.acme;

public class NormalEquationSolver {
    private int iteration;

    public double[] solveStep(double[] target, double[] start, double[][] jacobian, Matrix weight, Checker checker) {
        if (target.length != jacobian.length) { throw new IllegalArgumentException(); }
        if (start.length != jacobian[0].length) { throw new IllegalArgumentException(); }
        double[] objective = weight.multiply(jacobian).solve(target);
        while (!checker.converged(iteration++, start, objective)) { return objective; }
        return start;
    }
}

interface Matrix { Matrix multiply(double[][] jacobian); double[] solve(double[] objective); }
interface Checker { boolean converged(int iteration, double[] previous, double[] current); }
""".strip(),
        ),
        (
            "CurveFitterCore",
            "fitResiduals",
            """
package com.acme;

public class CurveFitterCore {
    private double[] start;
    private double[] target;
    private Model model;
    private Jacobian jacobian;

    public CurveFitterCore startAt(double[] start) { this.start = start; return this; }
    public CurveFitterCore target(double[] target) { this.target = target; return this; }
    public CurveFitterCore callbacks(Model model, Jacobian jacobian) { this.model = model; this.jacobian = jacobian; return this; }

    public double[] fitResiduals() {
        double[] residual = model.value(start);
        double[][] matrix = jacobian.value(start);
        if (residual.length != target.length) { throw new IllegalArgumentException(); }
        if (matrix.length != target.length) { throw new IllegalArgumentException(); }
        for (int iteration = 0; iteration < 2; iteration++) { return residual; }
        return target;
    }
}

interface Model { double[] value(double[] point); }
interface Jacobian { double[][] value(double[] point); }
""".strip(),
        ),
    ]

    for class_name, method_name, source_text in cases:
        source = tmp_path / f"src/main/java/com/acme/{class_name}.java"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source_text, encoding="utf-8")

        ctx = build_type_aware_context(source, tmp_path, method_name=method_name)
        rendered = render_execution_capsules_section(
            ctx.execution_recipes,
            method_name=method_name,
            target_shape=ctx.type_shapes[class_name],
        )

        assert "[Execution Capsule Skeleton]" in rendered
        assert "numeric_rank_safe_fixture" in rendered
        assert method_name in rendered
        assert "dimension" in rendered
        assert "rank" in rendered
        assert "target/residual length R" in rendered or "R=2" in rendered
        assert "GaussNewtonOptimizer" not in rendered
        assert "FromXmlParser" not in rendered
        assert infer_capsule_required_call_order(ctx.execution_recipes, method_name=method_name) == []


def test_private_target_recipe_prefers_public_entry_and_parameter_name_guidance(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/TemplateProcessor.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.Map;

public class TemplateProcessor {
    public String process(String template, Method method, Object[] args) {
        return resolveTemplate(template, method, args);
    }

    private String resolveTemplate(String template, Method method, Object[] args) {
        if (args == null || args.length == 0) {
            return template;
        }
        Parameter[] parameters = method.getParameters();
        Map<String, Object> values = new HashMap<>();
        for (int i = 0; i < parameters.length; i++) {
            values.put(parameters[i].getName(), args[i]);
        }
        String result = template;
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            result = result.replace("${" + entry.getKey() + "}", entry.getValue().toString());
        }
        return result;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="resolveTemplate")
    rendered = render_execution_recipes_section(ctx.execution_recipes)
    receipt = render_target_receipt_contract(ctx.execution_recipes, method_name="resolveTemplate")

    assert "caller-entry" in rendered
    assert "process" in rendered
    assert "do not invoke the private method directly" in rendered
    assert "Parameter.getName" in rendered
    assert "arg0" in rendered
    assert "private_public_entry_path" in receipt


def test_schema_recipe_requires_inline_fixtures_and_optional_unwrap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    source = tmp_path / "src/main/java/com/acme/SchemaFactory.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.lang.reflect.Type;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public class SchemaFactory {
    public Schema buildSchemaFromType(Type type) {
        ObjectMapper mapper = new ObjectMapper();
        JavaType javaType = mapper.constructType(type);
        BeanDescription description = mapper.getSerializationConfig().introspect(javaType);
        Map<String, Schema> properties = new LinkedHashMap<>();
        for (Property property : description.findProperties()) {
            properties.put(property.name(), new Schema(Optional.of("STRING"), Optional.empty()));
        }
        return new Schema(Optional.of("OBJECT"), Optional.of(properties));
    }
}

record Schema(Optional<String> type, Optional<Map<String, Schema>> properties) {}
interface Property { String name(); }
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="buildSchemaFromType")
    rendered = render_execution_recipes_section(ctx.execution_recipes)
    receipt = render_target_receipt_contract(ctx.execution_recipes, method_name="buildSchemaFromType")

    assert "schema-type-fixture" in rendered
    assert "SimplePojo" in rendered
    assert "RecursivePojo" in rendered
    assert "enum" in rendered
    assert "Optional" in rendered
    assert ".orElseThrow()" in rendered
    assert "containsKey" in rendered
    assert "primitive/string/boolean/numeric" in receipt
    assert "not-null" not in rendered.lower() or "shallow not-null" in rendered.lower()


def test_token_parser_recipe_requires_case_length_and_next_token_matrix(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    source = tmp_path / "src/main/java/com/acme/PatternCompiler.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class PatternCompiler {
    public void compilePattern(Builder builder, String pattern) {
        int[] indexRef = new int[1];
        for (int i = 0; i < pattern.length(); i++) {
            indexRef[0] = i;
            String token = parseToken(pattern, indexRef);
            i = indexRef[0];
            int tokenLen = token.length();
            if (tokenLen == 0) {
                break;
            }
            char c = token.charAt(0);
            switch (c) {
            case 'A':
            case 'B':
                if (tokenLen == 2) {
                    if (i + 1 < pattern.length()) {
                        indexRef[0]++;
                        if (isNumericToken(parseToken(pattern, indexRef))) {
                            builder.shortNumeric(c);
                        }
                        indexRef[0]--;
                    }
                    builder.shortToken(c);
                } else {
                    builder.longToken(c, tokenLen);
                }
                break;
            case 'C':
                builder.literal(token);
                break;
            default:
                throw new IllegalArgumentException();
            }
        }
    }

    private String parseToken(String pattern, int[] indexRef) { return pattern.substring(indexRef[0], indexRef[0] + 1); }
    private boolean isNumericToken(String token) { return Character.isDigit(token.charAt(0)); }
}

interface Builder { void shortNumeric(char c); void shortToken(char c); void longToken(char c, int len); void literal(String token); }
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="compilePattern")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "token-parser-matrix" in rendered
    assert "'A'" in rendered
    assert "'B'" in rendered
    assert "'C'" in rendered
    assert "token length" in rendered
    assert "next-token numeric" in rendered
    assert "non-numeric" in rendered
    assert "default/error" in rendered


def test_token_path_recipe_uses_structural_signals_without_parser_names(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tatg.analysis.typeinfo.recipe_synthesizer._legacy_name_heuristics_enabled", lambda: True)
    source = tmp_path / "src/main/java/com/acme/ScannerCore.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public class ScannerCore {
    private static final int STATE_NONE = 0;
    private static final int STATE_DIGIT = 1;
    private static final int STATE_DOT = 2;
    private final char[] buffer = new char[128];
    private int pos;
    private int limit;

    public int classify() {
        int state = STATE_NONE;
        char c = buffer[pos];
        switch (c) {
            case '-': state = STATE_DIGIT; break;
            case '.': state = STATE_DOT; break;
            case '+': state = STATE_NONE; break;
            default: throw new IllegalArgumentException();
        }
        if (pos == limit || Character.isDigit(buffer[pos + 1])) {
            return state;
        }
        return STATE_NONE;
    }
}
""".strip(),
        encoding="utf-8",
    )

    ctx = build_type_aware_context(source, tmp_path, method_name="classify")
    rendered = render_execution_recipes_section(ctx.execution_recipes)

    assert "token-parser-matrix" in rendered
    assert "STATE_DIGIT" in rendered
    assert "Character.isDigit" in rendered
    assert "buffer boundary" in rendered or "boundary" in rendered


def test_method_decl_regex_matches_package_private_methods():
    """Package-private methods must be recognised, not silently skipped.

    The access modifier was mandatory here while branch_analysis.py:68 already
    treated it as optional. That divergence skipped 23 of the 165 benchmark
    targets (14%) -- including low scorers such as JsonReader.doPeek and
    ByteMachine.addEndOfMatch -- so they received no execution recipes at all.
    """

    from tatg.analysis.typeinfo.recipe_synthesizer import _METHOD_DECL_RE

    def name_of(code):
        m = _METHOD_DECL_RE.search(code)
        return m.group(2) if m else None

    # Package-private, the previously-skipped case.
    assert name_of("    int doPeek() throws IOException {") == "doPeek"
    assert name_of("    @Override\n    Token nextToken() {") == "nextToken"
    assert name_of("    <T> List<T> wrap(Class<T> c) {") == "wrap"

    # Explicit modifiers must keep working, and must not be mis-parsed as the
    # return type now that the modifier group is optional.
    assert name_of("    public void writeString(String s) {") == "writeString"
    assert name_of("    private static int find(Object k) {") == "find"
    assert name_of("    protected final synchronized Map get() {") == "get"
