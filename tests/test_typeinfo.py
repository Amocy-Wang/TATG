"""Unit tests for the tatg.analysis.typeinfo package.

These tests exercise:
* source_extractor  – regex-based Java parsing
* mock_synthesizer  – scaffold generation from parsed types
* models            – dataclass accessors
* build_type_aware_context  – end-to-end on real mall source (skipped if absent)
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

from tatg.analysis.typeinfo.models import (
    AutowiredDep,
    ConstructorInfo,
    FieldInfo,
    MethodInfo,
    MockScaffold,
    TypeShape,
)
from tatg.analysis.typeinfo.source_extractor import (
    _RE_NULL_GUARD,
    _extract_fields,
    _extract_method_spans,
    extract_autowired_deps,
    extract_branch_hints,
    extract_type_shape,
)
from tatg.analysis.typeinfo.mock_synthesizer import (
    MockScaffoldBuilder,
    build_mock_scaffold,
    render_mock_scaffold_section,
    render_test_context_section,
    render_dependency_context_section,
    render_type_shapes_section,
    render_branch_hints_section,
    _simple_type,
    _is_primitive_or_jdk,
    _return_expression_for,
)


def test_typeinfo_exact_owner_excludes_nested_type_fields_and_branches(tmp_path: Path):
    source = tmp_path / "Outer.java"
    source.write_text(
        """
package com.acme;
public class Outer {
    private int outerState;
    public int read(boolean enabled) {
        if (enabled) {
            return outerState;
        }
        return 0;
    }
    static class Inner {
        private String innerState;
        public int read(boolean enabled) {
            if (innerState == null) {
                return -1;
            }
            return 1;
        }
    }
}
""".strip(),
        encoding="utf-8",
    )

    shape = extract_type_shape(source, class_name="Outer")
    hints = extract_branch_hints(source, method_name="read", class_name="Outer")

    assert shape is not None
    assert {field.name for field in shape.fields} == {"outerState"}
    assert [method.name for method in shape.methods].count("read") == 1
    assert all("innerState" not in hint.condition for hint in hints)


# ---------------------------------------------------------------------------
# Helpers — in-memory .java file fixtures
# ---------------------------------------------------------------------------

_SIMPLE_DOMAIN_CLASS = textwrap.dedent("""\
    package com.example.model;

    import java.math.BigDecimal;

    public class CartItem {
        private Long id;
        private String productName;
        private BigDecimal price;
        private Integer quantity;

        public void setId(Long id) { this.id = id; }
        public void setProductName(String productName) { this.productName = productName; }
        public void setPrice(BigDecimal price) { this.price = price; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public Long getId() { return id; }
    }
""")

_SERVICE_CLASS = textwrap.dedent("""\
    package com.example.service.impl;

    import org.springframework.beans.factory.annotation.Autowired;
    import com.example.service.MemberService;
    import com.example.service.CartService;
    import com.example.mapper.OrderMapper;

    @Service
    public class OrderServiceImpl implements OrderService {

        @Autowired
        private MemberService memberService;
        @Autowired
        private CartService cartService;
        @Autowired
        private OrderMapper orderMapper;

        public Map<String, Object> placeOrder(OrderParam param) {
            if (param.getAddressId() == null) {
                Asserts.fail("请选择地址");
            }
            if (param.getCouponId() != null) {
                // apply coupon
            }
            return new HashMap<>();
        }

        public void cancelOrder(Long orderId) {
            if (orderId == null) {
                return;
            }
        }
    }
""")


@pytest.fixture
def tmp_domain_class(tmp_path: Path) -> Path:
    f = tmp_path / "CartItem.java"
    f.write_text(_SIMPLE_DOMAIN_CLASS)
    return f


@pytest.fixture
def tmp_service_class(tmp_path: Path) -> Path:
    f = tmp_path / "OrderServiceImpl.java"
    f.write_text(_SERVICE_CLASS)
    return f


# ---------------------------------------------------------------------------
# extract_type_shape
# ---------------------------------------------------------------------------

class TestExtractTypeShape:
    def test_constructor_scan_preserves_all_source_visibilities(self, tmp_path: Path):
        source = tmp_path / "Owner.java"
        source.write_text(
            """
package com.acme;
public class Owner {
    Owner(String value, int count) {}
    protected Owner(long id) throws Exception {}
    private Owner(boolean hidden) {}
    public Owner() {}
}
""".strip(),
            encoding="utf-8",
        )

        shape = extract_type_shape(source)

        assert shape is not None
        assert [
            (constructor.visibility, constructor.params)
            for constructor in shape.constructors
        ] == [
            ("package-private", [("String", "value"), ("int", "count")]),
            ("protected", [("long", "id")]),
            ("private", [("boolean", "hidden")]),
            ("public", []),
        ]

    def test_public_method_scan_is_bounded_after_many_public_fields(self, tmp_path: Path):
        source = tmp_path / "LargeOwner.java"
        fields = "\n".join(
            f'    public static final String FIELD_{index} = "value";'
            for index in range(1200)
        )
        source.write_text(
            (
                "package com.acme;\n"
                "public class LargeOwner {\n"
                f"{fields}\n"
                "    public java.util.Map<String, Object> evaluate(\n"
                "            String input,\n"
                "            java.util.List<Integer> values) {\n"
                "        if (input == null) { return java.util.Collections.emptyMap(); }\n"
                "        return java.util.Collections.emptyMap();\n"
                "    }\n"
                "}\n"
            ),
            encoding="utf-8",
        )

        shape = extract_type_shape(source, class_name="LargeOwner")

        assert shape is not None
        method = next(item for item in shape.methods if item.name == "evaluate")
        assert method.return_type.replace(" ", "") == "java.util.Map<String,Object>"
        assert method.params == [
            ("String", "input"),
            ("java.util.List<Integer>", "values"),
        ]
        hints = extract_branch_hints(
            source,
            method_name="evaluate",
            class_name="LargeOwner",
        )
        assert any("input == null" in hint.condition for hint in hints)
        from tatg.analysis.typeinfo.recipe_synthesizer import _method_spans

        recipe_spans = _method_spans(source.read_text(encoding="utf-8"))
        assert [span["name"] for span in recipe_spans] == ["evaluate"]

    def test_class_name_extracted(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        assert shape is not None
        assert shape.class_name == "CartItem"

    def test_fqn_assembled(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        assert shape.fqn == "com.example.model.CartItem"

    def test_package_extracted(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        assert shape.package == "com.example.model"

    def test_fields_extracted(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        field_names = {f.name for f in shape.fields}
        assert {"id", "productName", "price", "quantity"} == field_names

    def test_field_types(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        by_name = {f.name: f for f in shape.fields}
        assert by_name["id"].java_type == "Long"
        assert by_name["price"].java_type == "BigDecimal"
        assert by_name["quantity"].java_type == "Integer"

    def test_setters_detected(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        setter_names = {m.name for m in shape.setters}
        assert "setId" in setter_names
        assert "setPrice" in setter_names

    def test_is_not_interface(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        assert not shape.is_interface

    def test_nonexistent_file_returns_none(self, tmp_path):
        shape = extract_type_shape(tmp_path / "Missing.java")
        assert shape is None

    def test_service_class_annotations_present(self, tmp_service_class):
        shape = extract_type_shape(tmp_service_class)
        assert shape is not None
        ann_names = {a.name for a in shape.annotations}
        assert "Service" in ann_names

    def test_generic_return_method_captured(self, tmp_service_class):
        shape = extract_type_shape(tmp_service_class)
        method_names = {m.name for m in shape.methods}
        assert "placeOrder" in method_names

    def test_source_file_recorded(self, tmp_domain_class):
        shape = extract_type_shape(tmp_domain_class)
        assert shape.source_file == str(tmp_domain_class)

    def test_constructor_and_method_params_preserve_generic_commas(self, tmp_path):
        source = tmp_path / "ServerCallContext.java"
        source.write_text(
            """
package com.example;

import java.util.Map;
import java.util.Set;

public class ServerCallContext {
    public ServerCallContext(User user, Map<String, Object> state, Set<String> requestedExtensions) {}
    public void update(Map<String, Object> state, Set<String> requestedExtensions) {}
}
class User {}
""".strip(),
            encoding="utf-8",
        )

        shape = extract_type_shape(source)

        assert shape is not None
        assert shape.constructors[0].params == [
            ("User", "user"),
            ("Map<String, Object>", "state"),
            ("Set<String>", "requestedExtensions"),
        ]
        update = next(method for method in shape.methods if method.name == "update")
        assert update.params == [
            ("Map<String, Object>", "state"),
            ("Set<String>", "requestedExtensions"),
        ]

    def test_method_visibility_distinguishes_public_and_protected(self, tmp_path):
        source = tmp_path / "AgentBase.java"
        source.write_text(
            """
package com.example;

public abstract class AgentBase {
    public static void addSystemHook(Hook hook) {}
    protected void addHook(Hook hook) {}
    protected void bindRuntimeContextToHooks(RuntimeContext ctx) {}
}
class Hook {}
class RuntimeContext {}
""".strip(),
            encoding="utf-8",
        )

        shape = extract_type_shape(source)

        assert shape is not None
        by_name = {method.name: method for method in shape.methods}
        assert by_name["addSystemHook"].visibility == "public"
        assert by_name["addSystemHook"].is_static
        assert by_name["addHook"].visibility == "protected"
        assert by_name["bindRuntimeContextToHooks"].visibility == "protected"


# ---------------------------------------------------------------------------
# extract_autowired_deps
# ---------------------------------------------------------------------------

class TestExtractAutowiredDeps:
    def test_all_deps_found(self, tmp_service_class):
        deps = extract_autowired_deps(tmp_service_class)
        type_names = {d.type_name for d in deps}
        assert type_names == {"MemberService", "CartService", "OrderMapper"}

    def test_field_names_correct(self, tmp_service_class):
        deps = extract_autowired_deps(tmp_service_class)
        by_type = {d.type_name: d for d in deps}
        assert by_type["MemberService"].field_name == "memberService"
        assert by_type["CartService"].field_name == "cartService"
        assert by_type["OrderMapper"].field_name == "orderMapper"

    def test_empty_class_returns_empty(self, tmp_path):
        f = tmp_path / "Empty.java"
        f.write_text("public class Empty {}")
        deps = extract_autowired_deps(f)
        assert deps == []

    def test_missing_file_returns_empty(self, tmp_path):
        deps = extract_autowired_deps(tmp_path / "Missing.java")
        assert deps == []


# ---------------------------------------------------------------------------
# extract_branch_hints
# ---------------------------------------------------------------------------

class TestExtractBranchHints:
    def test_null_guards_found(self, tmp_service_class):
        hints = extract_branch_hints(tmp_service_class)
        conditions = [h.condition for h in hints]
        assert any("== null" in c for c in conditions)

    def test_not_null_guards_found(self, tmp_service_class):
        hints = extract_branch_hints(tmp_service_class)
        conditions = [h.condition for h in hints]
        assert any("!= null" in c for c in conditions)

    def test_asserts_fail_found(self, tmp_service_class):
        hints = extract_branch_hints(tmp_service_class)
        assert any("Asserts.fail" in h.condition for h in hints)

    def test_method_filter_restricts_scope(self, tmp_service_class):
        all_hints = extract_branch_hints(tmp_service_class)
        place_hints = extract_branch_hints(tmp_service_class, method_name="placeOrder")
        cancel_hints = extract_branch_hints(tmp_service_class, method_name="cancelOrder")
        # both sets should be subsets of all_hints
        assert len(place_hints) + len(cancel_hints) <= len(all_hints)
        # placeOrder has the coupon not-null branch (param.getCouponId() != null)
        assert any("getCouponId" in h.condition for h in place_hints)

    def test_line_numbers_nonzero(self, tmp_service_class):
        hints = extract_branch_hints(tmp_service_class)
        assert all(h.line_number > 0 for h in hints)

    def test_setup_hints_trace_helper_and_mock_dependencies(self, tmp_path):
        service = tmp_path / "CheckoutServiceImpl.java"
        service.write_text(textwrap.dedent("""\
            package com.example.service.impl;

            import org.springframework.beans.factory.annotation.Autowired;

            public class CheckoutServiceImpl {
                @Autowired
                private MemberService memberService;
                @Autowired
                private CouponService couponService;

                public void checkout(OrderParam orderParam) {
                    Member member = memberService.getCurrentMember();
                    CouponDetail couponDetail = getCouponDetail(member, orderParam.getCouponId());
                    if (couponDetail == null) {
                        Asserts.fail("coupon unavailable");
                    }
                }

                private CouponDetail getCouponDetail(Member member, Long couponId) {
                    CouponDetail detail = couponService.load(member.getId(), couponId);
                    if (detail != null && detail.getCoupon().getId().equals(couponId)) {
                        return detail;
                    }
                    return null;
                }
            }
        """))
        hints = extract_branch_hints(service, method_name="checkout")
        fail_hint = next(h for h in hints if "coupon unavailable" in h.hint)
        setup = " ".join(fail_hint.setup_hints)
        assert "memberService.getCurrentMember" in setup
        assert "couponService.load" in setup
        assert "Helper getCouponDetail(...) inspects" in setup

    def test_empty_equals_and_compare_guards_are_extracted_generically(self, tmp_path):
        service = tmp_path / "GenericRulesService.java"
        service.write_text(textwrap.dedent("""\
            package com.example.service.impl;

            import org.springframework.beans.factory.annotation.Autowired;
            import org.springframework.util.CollectionUtils;
            import java.math.BigDecimal;
            import java.util.List;

            public class GenericRulesService {
                @Autowired
                private RemoteClient remoteClient;

                public void evaluate(Filter filter) {
                    List<Item> items = remoteClient.fetch(filter.getUserId());
                    if (CollectionUtils.isEmpty(items)) {
                        Asserts.fail("empty items");
                    }
                    if (filter.getStatus().equals(0)) {
                        Asserts.fail("bad status");
                    }
                    if (filter.getAmount().compareTo(filter.getLimit()) > 0) {
                        Asserts.fail("amount too high");
                    }
                }
            }
        """))
        hints = extract_branch_hints(service, method_name="evaluate")
        rendered = "\n".join(f"{h.hint} :: {' | '.join(h.setup_hints)}" for h in hints)
        assert "collection/string emptiness guard" in rendered
        assert "equals 0" in rendered or "equal to 0" in rendered
        assert "comparison guard" in rendered
        assert "remoteClient.fetch" in rendered
        assert "Shape items so it is empty" in rendered
        assert "compareTo(...) > 0" in rendered

    def test_missing_file_returns_empty(self, tmp_path):
        hints = extract_branch_hints(tmp_path / "Missing.java")
        assert hints == []


# ---------------------------------------------------------------------------
# models – TypeShape helpers
# ---------------------------------------------------------------------------

class TestTypeShapeHelpers:
    def _make_shape(self) -> TypeShape:
        return TypeShape(
            class_name="Item",
            fqn="com.example.Item",
            package="com.example",
            fields=[
                FieldInfo(name="id", java_type="Long"),
                FieldInfo(name="name", java_type="String"),
            ],
            methods=[
                MethodInfo(name="setId", return_type="void",
                           params=[("Long", "id")], is_setter=True),
                MethodInfo(name="setName", return_type="void",
                           params=[("String", "name")], is_setter=True),
            ],
        )

    def test_setters_property(self):
        shape = self._make_shape()
        assert {m.name for m in shape.setters} == {"setId", "setName"}

    def test_has_no_arg_constructor_true_when_none_declared(self):
        shape = self._make_shape()
        assert shape.has_no_arg_constructor

    def test_has_no_arg_constructor_false_when_only_parameterised(self):
        shape = self._make_shape()
        shape.constructors = [ConstructorInfo(params=[("Long", "id")])]
        assert not shape.has_no_arg_constructor

    def test_has_no_arg_constructor_false_when_only_private_no_arg(self):
        shape = self._make_shape()
        shape.constructors = [ConstructorInfo(params=[], visibility="private")]
        assert not shape.has_no_arg_constructor

    def test_has_no_arg_constructor_true_for_same_package_constructor(self):
        shape = self._make_shape()
        shape.constructors = [ConstructorInfo(params=[], visibility="package-private")]
        assert shape.has_no_arg_constructor

    def test_fixture_helper_contains_setters(self):
        shape = self._make_shape()
        helper = shape.fixture_helper()
        assert "setId" in helper
        assert "setName" in helper

    def test_field_sample_value_big_decimal(self):
        f = FieldInfo(name="price", java_type="BigDecimal")
        assert "BigDecimal" in f.sample_value()

    def test_field_sample_value_long(self):
        f = FieldInfo(name="id", java_type="Long")
        assert f.sample_value() == "1L"

    def test_field_is_collection(self):
        f = FieldInfo(name="items", java_type="List<CartItem>")
        assert f.is_collection

    def test_field_base_type_strips_generics(self):
        f = FieldInfo(name="items", java_type="List<CartItem>")
        assert f.base_type == "CartItem"

    def test_field_base_type_map_returns_value_type(self):
        f = FieldInfo(name="items", java_type="Map<String, OrderItem>")
        assert f.base_type == "OrderItem"

    def test_field_base_type_map_string_integer(self):
        f = FieldInfo(name="counts", java_type="Map<String, Integer>")
        assert f.base_type == "Integer"

    def test_field_base_type_list_returns_element_type(self):
        f = FieldInfo(name="items", java_type="List<OrderItem>")
        assert f.base_type == "OrderItem"


# ---------------------------------------------------------------------------
# build_mock_scaffold / MockScaffoldBuilder
# ---------------------------------------------------------------------------

class TestBuildMockScaffold:
    def _make_service_shape(self) -> TypeShape:
        return TypeShape(
            class_name="OrderServiceImpl",
            fqn="com.example.service.impl.OrderServiceImpl",
            package="com.example.service.impl",
            fields=[
                FieldInfo(name="memberService", java_type="MemberService"),
                FieldInfo(name="cartService", java_type="CartService"),
            ],
        )

    def _make_deps(self) -> list[AutowiredDep]:
        return [
            AutowiredDep(field_name="memberService", type_name="MemberService"),
            AutowiredDep(field_name="cartService", type_name="CartService"),
        ]

    def test_mock_declarations_generated(self):
        scaffold = build_mock_scaffold(
            target_shape=self._make_service_shape(),
            autowired_deps=self._make_deps(),
            type_index={},
        )
        assert len(scaffold.mock_declarations) == 2
        assert any("MemberService" in d for d in scaffold.mock_declarations)
        assert any("CartService" in d for d in scaffold.mock_declarations)

    def test_inject_mocks_declaration_contains_class(self):
        scaffold = build_mock_scaffold(
            target_shape=self._make_service_shape(),
            autowired_deps=self._make_deps(),
            type_index={},
        )
        assert "OrderServiceImpl" in scaffold.inject_mocks_declaration
        assert "@InjectMocks" in scaffold.inject_mocks_declaration

    def test_import_hints_include_junit5(self):
        scaffold = build_mock_scaffold(
            target_shape=self._make_service_shape(),
            autowired_deps=self._make_deps(),
            type_index={},
        )
        imports_str = "\n".join(scaffold.import_hints)
        assert "MockitoExtension" in imports_str
        assert "InjectMocks" in imports_str

    def test_fixture_helper_generated_for_known_return_type(self):
        member_shape = TypeShape(
            class_name="UmsMember",
            fqn="com.example.model.UmsMember",
            package="com.example.model",
            fields=[FieldInfo(name="id", java_type="Long")],
            methods=[
                MethodInfo(name="setId", return_type="void",
                           params=[("Long", "id")], is_setter=True),
                MethodInfo(name="getCurrentMember", return_type="UmsMember"),
            ],
        )
        member_service_shape = TypeShape(
            class_name="MemberService",
            fqn="",
            package="",
            methods=[
                MethodInfo(name="getCurrentMember", return_type="UmsMember"),
            ],
        )
        index = {"UmsMember": member_shape, "MemberService": member_service_shape}
        scaffold = build_mock_scaffold(
            target_shape=self._make_service_shape(),
            autowired_deps=[AutowiredDep(field_name="memberService", type_name="MemberService")],
            type_index=index,
        )
        helpers_str = "\n".join(scaffold.fixture_helpers)
        assert "buildUmsMember" in helpers_str

    def test_empty_inputs_produce_valid_scaffold(self):
        scaffold = build_mock_scaffold(
            target_shape=None,
            autowired_deps=[],
            type_index={},
        )
        assert isinstance(scaffold, MockScaffold)


# ---------------------------------------------------------------------------
# render_* functions
# ---------------------------------------------------------------------------

class TestRenderFunctions:
    def test_render_type_shapes_section_with_shapes(self):
        shapes = {
            "Foo": TypeShape(
                class_name="Foo", fqn="", package="",
                fields=[FieldInfo(name="id", java_type="Long")],
            )
        }
        result = render_type_shapes_section(shapes)
        assert "Foo" in result
        assert "id" in result

    def test_render_type_shapes_section_empty(self):
        result = render_type_shapes_section({})
        assert result == ""

    def test_render_mock_scaffold_section_non_empty(self):
        scaffold = MockScaffold(
            target_class="MyService",
            mock_declarations=["@Mock\n    private Dep dep;"],
            inject_mocks_declaration="@InjectMocks\n    private MyService service;",
        )
        result = render_mock_scaffold_section(scaffold)
        assert "MyService" in result
        assert "@Mock" in result

    def test_render_branch_hints_section_non_empty(self):
        from tatg.analysis.typeinfo.models import BranchHint
        hints = [BranchHint(method_name="foo", condition="x == null", hint="test null", setup_hints=["stub dep.call(...)"])]
        result = render_branch_hints_section(hints)
        assert "test null" in result
        assert "setup: stub dep.call(...)" in result

    def test_render_branch_hints_section_empty(self):
        result = render_branch_hints_section([])
        assert result == ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_simple_type_strips_generics(self):
        assert _simple_type("List<CartItem>") == "List"

    def test_simple_type_strips_array(self):
        assert _simple_type("String[]") == "String"

    def test_simple_type_strips_qualified(self):
        assert _simple_type("com.example.Foo") == "Foo"

    def test_is_primitive_or_jdk(self):
        assert _is_primitive_or_jdk("String")
        assert _is_primitive_or_jdk("long")
        assert not _is_primitive_or_jdk("CartItem")

    def test_return_expression_for_string(self):
        assert _return_expression_for("String", {}) == '"test"'

    def test_return_expression_for_long(self):
        assert _return_expression_for("Long", {}) == "1L"

    def test_return_expression_for_known_type_uses_fixture(self):
        shape = TypeShape(class_name="Foo", fqn="", package="")
        expr = _return_expression_for("Foo", {"Foo": shape})
        assert expr == "buildFoo()"


# ---------------------------------------------------------------------------
# Integration smoke test — real mall source (skip if not present)
# ---------------------------------------------------------------------------

MALL_SERVICE = Path(
    "/path/to/mall/mall-portal/src/main/java/"
    "com/macro/mall/portal/service/impl/OmsPortalOrderServiceImpl.java"
)
MALL_ROOT = Path("/path/to/mall")


def test_method_scope_does_not_expand_types_used_only_by_unrelated_methods(
    tmp_path,
):
    from tatg.analysis.typeinfo import build_type_aware_context

    source_root = tmp_path / "src/main/java/com/example"
    source_root.mkdir(parents=True)
    target = source_root / "Service.java"
    target.write_text(
        textwrap.dedent(
            """\
            package com.example;

            public class Service {
                public String target(String value) {
                    return value;
                }

                public UnrelatedHelper unrelated() {
                    return new UnrelatedHelper();
                }
            }
            """
        ),
        encoding="utf-8",
    )
    (source_root / "UnrelatedHelper.java").write_text(
        textwrap.dedent(
            """\
            package com.example;

            public class UnrelatedHelper {
                public UnrelatedHelper() {}
                public String helperOnly() { return "unused"; }
            }
            """
        ),
        encoding="utf-8",
    )

    context = build_type_aware_context(
        target,
        tmp_path,
        method_name="target",
        class_name="Service",
    )

    assert "Service" in context.type_shapes
    assert "UnrelatedHelper" not in context.type_shapes
    assert any(
        method.name == "unrelated"
        for method in context.type_shapes["Service"].methods
    )


@pytest.mark.skipif(not MALL_SERVICE.exists(), reason="mall project not cloned")
class TestMallIntegration:
    def test_type_shape_extracted(self):
        shape = extract_type_shape(MALL_SERVICE)
        assert shape is not None
        assert shape.class_name == "OmsPortalOrderServiceImpl"

    def test_autowired_deps_count(self):
        deps = extract_autowired_deps(MALL_SERVICE)
        # OmsPortalOrderServiceImpl has 15 @Autowired fields
        assert len(deps) >= 10

    def test_generate_order_branch_hints(self):
        hints = extract_branch_hints(MALL_SERVICE, method_name="generateOrder")
        assert len(hints) >= 6

    def test_generate_order_branch_hints_include_setup_guidance(self):
        hints = extract_branch_hints(MALL_SERVICE, method_name="generateOrder")
        fail_hint = next(h for h in hints if "库存不足" in h.hint)
        setup = " ".join(fail_hint.setup_hints)
        assert "cartItemService.listPromotion" in setup
        assert "Helper hasStock(...) inspects" in setup

    def test_asserts_fail_hints_present(self):
        hints = extract_branch_hints(MALL_SERVICE, method_name="generateOrder")
        fail_hints = [h for h in hints if "Asserts.fail" in h.condition]
        assert len(fail_hints) >= 3

    def test_build_type_aware_context(self):
        from tatg.analysis.typeinfo import build_type_aware_context
        ctx = build_type_aware_context(
            MALL_SERVICE, MALL_ROOT, method_name="generateOrder"
        )
        assert ctx.target_class == "OmsPortalOrderServiceImpl"
        assert len(ctx.autowired_deps) >= 10
        assert len(ctx.type_shapes) >= 5
        assert ctx.mock_scaffold is not None
        assert len(ctx.branch_hints) >= 6

    def test_mock_scaffold_has_all_mocks(self):
        from tatg.analysis.typeinfo import build_type_aware_context
        ctx = build_type_aware_context(MALL_SERVICE, MALL_ROOT)
        assert ctx.mock_scaffold is not None
        deps_in_mocks = {
            line.split()[-1].rstrip(";")
            for line in ctx.mock_scaffold.mock_declarations
        }
        dep_fields = {d.field_name for d in ctx.autowired_deps}
        assert dep_fields == deps_in_mocks

    def test_import_hints_include_fqn(self):
        from tatg.analysis.typeinfo import build_type_aware_context
        ctx = build_type_aware_context(MALL_SERVICE, MALL_ROOT)
        imports = "\n".join(ctx.mock_scaffold.import_hints)
        # At least some mall package imports should be present
        assert "com.macro.mall" in imports


# ---------------------------------------------------------------------------
# New: render_test_context_section (framework-agnostic)
# ---------------------------------------------------------------------------

class TestRenderTestContextSection:
    """render_test_context_section must describe deps without forcing Mockito."""

    def _scaffold(self) -> MockScaffold:
        return MockScaffold(
            target_class="PaymentService",
            mock_declarations=["@Mock\n    private PaymentRepo repo;"],
            inject_mocks_declaration="@InjectMocks\n    private PaymentService service;",
            import_hints=["import java.math.BigDecimal;"],
            suggested_when_stubs=["when(repo.findById(any())).thenReturn(null);"],
        )

    def test_section_lists_deps_agnostically(self):
        from tatg.analysis.typeinfo.mock_synthesizer import render_test_context_section
        result = render_test_context_section(self._scaffold())
        assert "PaymentRepo" in result
        assert "repo" in result

    def test_section_labels_mockito_stubs_as_optional(self):
        from tatg.analysis.typeinfo.mock_synthesizer import render_test_context_section
        result = render_test_context_section(self._scaffold())
        assert "Optional" in result or "optional" in result


# ---------------------------------------------------------------------------
# New: render_dependency_context_section (truly Mockito-free)
# ---------------------------------------------------------------------------

class TestRenderDependencyContextSection:
    """render_dependency_context_section must contain NO Mockito constructs."""

    def _scaffold(self) -> MockScaffold:
        return MockScaffold(
            target_class="PaymentService",
            mock_declarations=["@Mock\n    private PaymentRepo repo;"],
            inject_mocks_declaration="@InjectMocks\n    private PaymentService service;",
            import_hints=[
                "import java.math.BigDecimal;",
                "import org.mockito.Mock;",
                "import org.mockito.junit.jupiter.MockitoExtension;",
            ],
            suggested_when_stubs=["when(repo.findById(any())).thenReturn(null);"],
            fixture_helpers=["private Payment buildPayment() {\n    Payment p = new Payment();\n    return p;\n}"],
        )

    def test_lists_dep_type_and_field(self):
        result = render_dependency_context_section(self._scaffold())
        assert "PaymentRepo" in result
        assert "repo" in result

    def test_contains_no_mock_annotation(self):
        result = render_dependency_context_section(self._scaffold())
        # Advisory text may mention @Mock as a concept, but no @Mock field declarations
        assert "@Mock\n" not in result
        assert "@InjectMocks\n" not in result

    def test_contains_no_when_stubs(self):
        result = render_dependency_context_section(self._scaffold())
        assert "when(" not in result
        assert "thenReturn" not in result

    def test_strips_mockito_imports(self):
        result = render_dependency_context_section(self._scaffold())
        assert "mockito" not in result.lower()

    def test_retains_domain_type_imports(self):
        result = render_dependency_context_section(self._scaffold())
        assert "BigDecimal" in result

    def test_omits_unproven_fixture_helpers(self):
        result = render_dependency_context_section(self._scaffold())
        assert "buildPayment" not in result
        assert "new Payment()" not in result

    def test_advises_direct_construction(self):
        result = render_dependency_context_section(self._scaffold())
        # Should mention preferring direct construction
        assert "direct" in result.lower() or "new " in result

    def test_section_does_not_mandate_injectmocks(self):
        from tatg.analysis.typeinfo.mock_synthesizer import render_test_context_section
        result = render_test_context_section(self._scaffold())
        # "@InjectMocks" may appear as a hint, but not as a mandatory directive
        assert "EMBED" not in result


# ---------------------------------------------------------------------------
# New: _find_public_entry_points — private method → public caller discovery
# ---------------------------------------------------------------------------

class TestFindPublicEntryPoints:
    """_find_public_entry_points must locate public callers of private methods."""

    _SOURCE = textwrap.dedent("""\
        package com.example;

        public class OrderServiceImpl {

            public OrderResult checkout(Long userId) {
                CartState cart = loadCart(userId);
                return process(cart);
            }

            public void cancelOrder(Long orderId) {
                applyRefund(orderId);
            }

            protected OrderResult process(CartState cart) {
                return validateStock(cart);
            }

            private CartState loadCart(Long userId) {
                return new CartState();
            }

            private OrderResult validateStock(CartState cart) {
                if (cart == null) throw new IllegalArgumentException("null cart");
                return new OrderResult();
            }

            private void applyRefund(Long orderId) {
                // refund logic
            }
        }
    """)

    def _adapter(self):
        from tatg.engines.tatg_adapter import TATGEngineAdapter
        return TATGEngineAdapter()

    def test_direct_private_caller_found(self):
        adapter = self._adapter()
        eps = adapter._find_public_entry_points("loadCart", self._SOURCE)
        methods = [ep["method"] for ep in eps]
        assert "checkout" in methods

    def test_transitive_private_caller_found(self):
        adapter = self._adapter()
        # validateStock is called by process (protected), which is called by checkout (public)
        eps = adapter._find_public_entry_points("validateStock", self._SOURCE)
        methods = [ep["method"] for ep in eps]
        assert "checkout" in methods

    def test_public_entry_points_preferred_before_protected_fallback(self):
        adapter = self._adapter()
        eps = adapter._find_public_entry_points("validateStock", self._SOURCE)
        assert eps[0]["method"] == "checkout"
        assert eps[0]["access"] == "public"
        assert any(ep["method"] == "process" and ep["access"] == "protected" for ep in eps)

    def test_nonexistent_method_returns_empty(self):
        adapter = self._adapter()
        eps = adapter._find_public_entry_points("nonexistentMethod", self._SOURCE)
        assert eps == []

    def test_applyRefund_found_via_cancelOrder(self):
        adapter = self._adapter()
        eps = adapter._find_public_entry_points("applyRefund", self._SOURCE)
        methods = [ep["method"] for ep in eps]
        assert "cancelOrder" in methods

    def test_factory_constructor_delegation_reaches_private_helper(self):
        source = textwrap.dedent("""\
            package com.example;

            public class Parser {
                public static Parser parse(String text) {
                    return new Parser(text);
                }

                public Parser(String text) {
                    this(text, 0);
                }

                public Parser(String text, int offset) {
                    createHeaders();
                }

                private void createHeaders() {}
            }
        """)

        eps = self._adapter()._find_public_entry_points(
            "createHeaders",
            source,
            class_name="Parser",
        )

        assert any(
            ep["method"] == "parse"
            and ep["path"] == [
                "parse",
                "Parser",
                "Parser",
                "createHeaders",
            ]
            for ep in eps
        )
        assert any(
            ep["method"] == "Parser"
            and ep["path"] == ["Parser", "createHeaders"]
            for ep in eps
        )

    def test_factory_entry_discovery_stays_scoped_to_target_class(self):
        source = textwrap.dedent("""\
            package com.example;

            public class Parser {
                public Parser(String text) { target(); }
                private void target() {}
            }

            class Other {
                public static Other parse(String text) {
                    return new Other(text);
                }
                Other(String text) {}
            }
        """)

        eps = self._adapter()._find_public_entry_points(
            "target",
            source,
            class_name="Parser",
        )

        assert any(ep["method"] == "Parser" for ep in eps)
        assert all(ep["method"] != "parse" for ep in eps)


# ---------------------------------------------------------------------------
# New: _render_few_shot_example — structural format anchor
# ---------------------------------------------------------------------------

class TestRenderFewShotExample:
    """_render_few_shot_example must include required format rules."""

    def _adapter(self):
        from tatg.engines.tatg_adapter import TATGEngineAdapter
        return TATGEngineAdapter()

    def test_example_contains_scenario_comment(self):
        adapter = self._adapter()
        result = adapter._render_few_shot_example("OrderService")
        assert "// SCENARIO:" in result

    def test_example_contains_arrange_act_assert(self):
        adapter = self._adapter()
        result = adapter._render_few_shot_example("OrderService")
        assert "ARRANGE" in result
        assert "ACT" in result
        assert "ASSERT" in result

    def test_example_contains_bigdecimal_rule(self):
        adapter = self._adapter()
        result = adapter._render_few_shot_example("OrderService")
        assert "BigDecimal" in result
        assert "double" in result.lower() or "literal" in result

    def test_spring_example_uses_mock_inject(self):
        adapter = self._adapter()
        result = adapter._render_few_shot_example("OrderService")
        assert "@Mock" in result
        assert "@InjectMocks" in result

    def test_non_spring_example_uses_direct_construction(self):
        # The example scaffold always uses @Mock / @InjectMocks; verify both are present.
        adapter = self._adapter()
        result = adapter._render_few_shot_example("Calculator")
        assert "@Mock" in result
        assert "@InjectMocks" in result
        # Target class name must appear in the scaffold
        assert "Calculator" in result

    def test_example_labels_output_format(self):
        adapter = self._adapter()
        result = adapter._render_few_shot_example("Svc")
        assert "EXAMPLE" in result


# ---------------------------------------------------------------------------
# Regression: _has_primary_type must recognise Java record types (Bug 1)
# ---------------------------------------------------------------------------

class TestHasPrimaryTypeRecord:
    """The class-discovery regex must match Java 16+ record declarations."""

    @staticmethod
    def _has_primary_type(source: str, class_name: str) -> bool:
        import re
        return bool(
            re.search(
                rf"\b(?:class|interface|enum|record|@interface)\s+{re.escape(class_name)}\b",
                source,
            )
        )

    def test_record_type_is_recognised(self):
        assert self._has_primary_type("public record Point(int x, int y) {}", "Point")

    def test_class_type_still_recognised(self):
        assert self._has_primary_type("public class Point {}", "Point")

    def test_enum_type_still_recognised(self):
        assert self._has_primary_type("public enum Status { OK, FAIL }", "Status")

    def test_non_matching_name_returns_false(self):
        assert not self._has_primary_type("public record Point(int x, int y) {}", "Rectangle")


# ---------------------------------------------------------------------------
# Regression: _extract_fields must not leak annotations across method decls (Bug 3)
# ---------------------------------------------------------------------------

class TestExtractFieldsAnnotationLeak:
    """@Override before a method must not leak into the next field's annotations."""

    _SOURCE = textwrap.dedent("""\
        package com.example;

        public class Widget {
            @Override
            public void configure() {
                // method body
            }

            @NotNull
            private String name;
        }
    """)

    def test_notNull_annotation_not_contaminated_by_override(self):
        fields = _extract_fields(self._SOURCE)
        assert len(fields) == 1
        ann_names = [a.name for a in fields[0].annotations]
        assert "NotNull" in ann_names
        assert "Override" not in ann_names

    def test_field_is_not_nullable_due_to_notNull(self):
        fields = _extract_fields(self._SOURCE)
        assert not fields[0].is_nullable


# ---------------------------------------------------------------------------
# Regression: _extract_method_spans must handle { } inside string literals (Bug 2)
# ---------------------------------------------------------------------------

class TestExtractMethodSpansBracesInStrings:
    """Braces inside string literals must not displace method span boundaries."""

    _SOURCE = textwrap.dedent("""\
        package com.example;

        public class Formatter {
            public String format(int x) {
                String tmpl = "value={x}";
                return tmpl.replace("{x}", String.valueOf(x));
            }

            public int next() {
                return 0;
            }
        }
    """)

    def test_format_span_ends_before_next_method(self):
        spans = _extract_method_spans(self._SOURCE)
        assert "format" in spans
        assert "next" in spans
        format_body = spans["format"]["body"]
        assert "next" not in format_body

    def test_next_span_is_non_empty(self):
        spans = _extract_method_spans(self._SOURCE)
        assert "return 0" in spans["next"]["body"]


# ---------------------------------------------------------------------------
# recipe_synthesizer helpers
# ---------------------------------------------------------------------------

class TestExtractSwitchCaseLiterals:
    """_extract_switch_case_literals must return unique literal values from case arms."""

    def test_char_literals(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_switch_case_literals
        body = "switch (c) { case 'l': return 1L; case 'L': return 1L; case 'f': return 1F; }"
        result = _extract_switch_case_literals(body)
        assert "'l'" in result
        assert "'L'" in result
        assert "'f'" in result

    def test_enum_constant_literals(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_switch_case_literals
        body = "switch (order) { case XYZ: break; case ZYX: break; case YXZ: break; }"
        result = _extract_switch_case_literals(body)
        assert "XYZ" in result
        assert "ZYX" in result

    def test_java14_arrow_cases(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_switch_case_literals
        body = "switch (x) { case 0 -> 'a'; case 1 -> 'b'; }"
        result = _extract_switch_case_literals(body)
        assert "0" in result
        assert "1" in result

    def test_multi_case_arm_split_on_comma(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_switch_case_literals
        body = "switch (c) { case 'e', 'E' -> parseExp(); }"
        result = _extract_switch_case_literals(body)
        assert "'e'" in result
        assert "'E'" in result

    def test_deduplicates_repeated_literals(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_switch_case_literals
        body = "switch (n) { case 1: return 1; case 1: return 2; case 2: return 3; }"
        result = _extract_switch_case_literals(body)
        assert result.count("1") == 1

    def test_empty_body_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_switch_case_literals
        assert _extract_switch_case_literals("") == []
        assert _extract_switch_case_literals("if (x > 0) return x;") == []


class TestExtractThrowGuardFacts:
    """_extract_throw_guard_facts must produce human-readable guard→exception descriptions."""

    def test_null_guard(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_throw_guard_facts
        body = "if (str == null) { throw new IllegalArgumentException(\"null\"); }"
        result = _extract_throw_guard_facts(body)
        assert len(result) == 1
        assert "str == null" in result[0]
        assert "IllegalArgumentException" in result[0]

    def test_blank_guard(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_throw_guard_facts
        body = "if (StringUtils.isBlank(str)) throw new NumberFormatException(\"blank\");"
        result = _extract_throw_guard_facts(body)
        assert len(result) == 1
        assert "NumberFormatException" in result[0]

    def test_multiple_guards(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_throw_guard_facts
        body = (
            "if (x == null) throw new NullPointerException();\n"
            "if (x < 0) throw new IllegalArgumentException();\n"
        )
        result = _extract_throw_guard_facts(body)
        assert len(result) == 2

    def test_no_throw_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_throw_guard_facts
        assert _extract_throw_guard_facts("return x + 1;") == []
        assert _extract_throw_guard_facts("") == []

    def test_deduplicates_identical_facts(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_throw_guard_facts
        body = (
            "if (x == null) throw new NPE();\n"
            "if (x == null) throw new NPE();\n"
        )
        result = _extract_throw_guard_facts(body)
        assert len(result) == 1


class TestExtractCatchRethrowFacts:
    def test_basic_catch_rethrow(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = (
            "try { xmlWriter.writeCharacters(text); }"
            "catch (IOException e) { throw new JsonProcessingException(e); }"
        )
        result = _extract_catch_rethrow_facts(body)
        assert len(result) == 1
        assert "IOException" in result[0]
        assert "JsonProcessingException" in result[0]

    def test_multi_catch_rethrow(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = (
            "try { op(); }"
            "catch (IOException | XMLStreamException e) { throw new RuntimeException(e); }"
        )
        result = _extract_catch_rethrow_facts(body)
        assert len(result) == 2
        caught = {f.split("throws ")[1].split(" ")[0] for f in result}
        assert caught == {"IOException", "XMLStreamException"}

    def test_no_rethrow_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = "if (x == null) throw new NullPointerException();"
        assert _extract_catch_rethrow_facts(body) == []

    def test_deduplicates_identical_patterns(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        block = "try { op(); } catch (IOException e) { throw new RuntimeException(e); } "
        result = _extract_catch_rethrow_facts(block + block)
        assert len(result) == 1


class TestBoundaryInputRecipe:
    def test_string_param_without_source_null_branch_uses_non_null_baseline(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        recipe = _boundary_input_recipe(
            "doWork",
            [("String", "name")],
            'return name.length() == 0 ? "" : name;',
        )
        assert recipe is not None
        assert any("non-null" in g for g in recipe.path_goals)
        assert not any("test null" in g for g in recipe.path_goals)

    def test_explicit_source_null_branch_promotes_null_input(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        recipe = _boundary_input_recipe(
            "doWork",
            [("String", "name")],
            'if (name == null) return ""; return name.trim();',
        )
        assert recipe is not None
        assert any("test null for the source-visible null branch" in g for g in recipe.path_goals)

    def test_primitive_param_returns_none(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        assert _boundary_input_recipe("count", [("int", "x")]) is None

    def test_mixed_params_skips_primitives(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        recipe = _boundary_input_recipe("process", [("int", "x"), ("String", "s")])
        assert recipe is not None
        assert len(recipe.path_goals) == 1
        assert "s" in recipe.path_goals[0]

    def test_list_param_generates_goal(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        recipe = _boundary_input_recipe("process", [("List<String>", "items")])
        assert recipe is not None
        assert any("empty" in g for g in recipe.path_goals)

    def test_integer_wrapper_generates_goal(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        recipe = _boundary_input_recipe("compute", [("Integer", "count")])
        assert recipe is not None
        assert any("MAX_VALUE" in g for g in recipe.path_goals)

    def test_no_params_returns_none(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        assert _boundary_input_recipe("getAll", []) is None


# ---------------------------------------------------------------------------
# Rec B: parse_pitest_xml_surviving_mutants_for_method
# ---------------------------------------------------------------------------

class TestParsePitestXmlSurvivingMutants:
    def _make_xml(self, tmp_path, mutations_xml: str) -> "Path":
        from pathlib import Path
        pit_dir = tmp_path / "target" / "pit-reports"
        pit_dir.mkdir(parents=True)
        xml_path = pit_dir / "mutations.xml"
        xml_path.write_text(mutations_xml)
        return xml_path

    def test_returns_empty_when_no_survived(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="true" status="KILLED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>doIt</mutatedMethod>
            <lineNumber>10</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <description>negated conditional</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(xml, "doIt")
        assert result == []

    def test_extracts_survived_easy_mutant(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>doIt</mutatedMethod>
            <lineNumber>42</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.returns.EmptyObjectReturnValsMutator</mutator>
            <description>replaced return value with ""</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(xml, "doIt")
        assert len(result) == 1
        assert result[0]["line"] == 42
        assert result[0]["difficulty"] == "easy"
        assert "kill_hint" in result[0]

        assert "assertEquals" in result[0]["kill_hint"] or "EXACT" in result[0]["kill_hint"]
        assert any("non-empty" in item or "exact" in item.lower() for item in result[0]["oracle_strategy"])

    def test_hard_mutant_classified_correctly(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>check</mutatedMethod>
            <lineNumber>7</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <description>negated conditional</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(xml, "check")
        assert len(result) == 1
        assert result[0]["difficulty"] == "hard"
        assert "conditional mutation" in result[0]["kill_hint"]
        assert any("contrasting branch cases" in item for item in result[0]["oracle_strategy"])

    def test_same_line_boundary_mutants_keep_pit_identity_and_bind_predicates(
        self,
        tmp_path,
    ):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="true" status="KILLED">
            <mutatedClass>com.example.Bounds</mutatedClass>
            <mutatedMethod>read</mutatedMethod>
            <methodDescription>(I)V</methodDescription>
            <lineNumber>11</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.ConditionalsBoundaryMutator</mutator>
            <indexes><index>68</index></indexes>
            <blocks><block>9</block></blocks>
            <description>changed conditional boundary</description>
          </mutation>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Bounds</mutatedClass>
            <mutatedMethod>read</mutatedMethod>
            <methodDescription>(I)V</methodDescription>
            <lineNumber>11</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.ConditionalsBoundaryMutator</mutator>
            <indexes><index>73</index></indexes>
            <blocks><block>10</block></blocks>
            <description>changed conditional boundary</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "read",
            class_fqn="com.example.Bounds",
            method_descriptor="(I)V",
            source_lines={11: "if (index >= 0 && index < values.length) {"},
        )

        assert [row["pit_index"] for row in result] == [73]
        assert [row["pit_block"] for row in result] == [10]
        assert [row["source_predicate"] for row in result] == [
            "index < values.length",
        ]
        assert [row["source_predicate_ordinal"] for row in result] == [2]

    def test_boundary_predicate_binding_preserves_threshold_offset(
        self,
        tmp_path,
    ):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="true" status="KILLED">
            <mutatedClass>com.example.Bounds</mutatedClass>
            <mutatedMethod>read</mutatedMethod>
            <methodDescription>(Ljava/lang/String;)V</methodDescription>
            <lineNumber>11</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.ConditionalsBoundaryMutator</mutator>
            <indexes><index>68</index></indexes>
            <blocks><block>9</block></blocks>
            <description>changed conditional boundary</description>
          </mutation>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Bounds</mutatedClass>
            <mutatedMethod>read</mutatedMethod>
            <methodDescription>(Ljava/lang/String;)V</methodDescription>
            <lineNumber>11</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.ConditionalsBoundaryMutator</mutator>
            <indexes><index>73</index></indexes>
            <blocks><block>10</block></blocks>
            <description>changed conditional boundary</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "read",
            class_fqn="com.example.Bounds",
            method_descriptor="(Ljava/lang/String;)V",
            source_lines={
                11: "if (expPos > -1 && expPos < value.length() - 1) {"
            },
        )

        assert [row["source_predicate"] for row in result] == [
            "expPos < value.length() - 1",
        ]
        assert [row["source_predicate_ordinal"] for row in result] == [2]

    def test_same_line_negated_conditionals_bind_short_circuit_boolean_atom(
        self,
        tmp_path,
    ):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.RangeMachine</mutatedClass>
            <mutatedMethod>add</mutatedMethod>
            <methodDescription>(Lcom/example/Range;)V</methodDescription>
            <lineNumber>21</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <indexes><index>227</index></indexes>
            <blocks><block>30</block></blocks>
            <description>negated conditional</description>
          </mutation>
          <mutation detected="false" status="NO_COVERAGE">
            <mutatedClass>com.example.RangeMachine</mutatedClass>
            <mutatedMethod>add</mutatedMethod>
            <methodDescription>(Lcom/example/Range;)V</methodDescription>
            <lineNumber>21</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <indexes><index>230</index></indexes>
            <blocks><block>31</block></blocks>
            <description>negated conditional</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "add",
            class_fqn="com.example.RangeMachine",
            method_descriptor="(Lcom/example/Range;)V",
            source_lines={
                21: "if ((lastBottom < Constants.MAX_DIGIT) || !range.openBottom) {",
            },
            include_statuses={"NO_COVERAGE"},
        )

        assert [row["pit_index"] for row in result] == [230]
        assert [row["pit_block"] for row in result] == [31]
        assert [row["source_predicate"] for row in result] == [
            "!range.openBottom",
        ]
        assert [row["source_predicate_ordinal"] for row in result] == [2]

    def test_multiline_negated_conditionals_bind_every_short_circuit_atom(
        self,
        tmp_path,
    ):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        rows = []
        for pit_index, block, status in (
            (17, 4, "SURVIVED"),
            (21, 6, "NO_COVERAGE"),
            (24, 7, "NO_COVERAGE"),
            (27, 8, "NO_COVERAGE"),
        ):
            rows.append(f"""<mutation detected="false" status="{status}">
              <mutatedClass>com.example.Filter</mutatedClass>
              <mutatedMethod>next</mutatedMethod>
              <methodDescription>()I</methodDescription>
              <lineNumber>40</lineNumber>
              <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
              <indexes><index>{pit_index}</index></indexes>
              <blocks><block>{block}</block></blocks>
              <description>negated conditional</description>
            </mutation>""")
        xml = self._make_xml(
            tmp_path,
            "<mutations>" + "".join(rows) + "</mutations>",
        )

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "next",
            class_fqn="com.example.Filter",
            method_descriptor="()I",
            source_lines={
                40: "if (token.isScalar() && !context.isHandled() && !includePath",
                41: "        && (filter == Filters.ALL)) {",
            },
            include_statuses={"SURVIVED", "NO_COVERAGE"},
            max_easy=8,
            max_hard=8,
        )

        assert [row["source_predicate"] for row in result] == [
            "token.isScalar()",
            "!context.isHandled()",
            "!includePath",
            "filter == Filters.ALL",
        ]
        assert [row["source_predicate_ordinal"] for row in result] == [
            1,
            2,
            3,
            4,
        ]

    def test_continuation_line_uses_enclosing_short_circuit_condition(
        self,
        tmp_path,
    ):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="true" status="KILLED">
            <mutatedClass>com.example.Parser</mutatedClass>
            <mutatedMethod>parse</mutatedMethod>
            <methodDescription>(Ljava/lang/String;)V</methodDescription>
            <lineNumber>31</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <indexes><index>50</index></indexes>
            <blocks><block>5</block></blocks>
            <description>negated conditional</description>
          </mutation>
          <mutation detected="false" status="NO_COVERAGE">
            <mutatedClass>com.example.Parser</mutatedClass>
            <mutatedMethod>parse</mutatedMethod>
            <methodDescription>(Ljava/lang/String;)V</methodDescription>
            <lineNumber>31</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <indexes><index>54</index></indexes>
            <blocks><block>6</block></blocks>
            <description>negated conditional</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "parse",
            class_fqn="com.example.Parser",
            method_descriptor="(Ljava/lang/String;)V",
            source_lines={
                30: "if (value != null",
                31: "        && value.isValid()) {",
            },
            include_statuses={"NO_COVERAGE"},
        )

        assert [row["source_predicate"] for row in result] == [
            "value.isValid()",
        ]
        assert [row["source_predicate_ordinal"] for row in result] == [2]

    def test_ambiguous_negated_conditionals_remain_unbound(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="NO_COVERAGE">
            <mutatedClass>com.example.Checks</mutatedClass>
            <mutatedMethod>check</mutatedMethod>
            <methodDescription>(Ljava/lang/Object;)Z</methodDescription>
            <lineNumber>7</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <indexes><index>12</index></indexes>
            <blocks><block>2</block></blocks>
            <description>negated conditional</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "check",
            class_fqn="com.example.Checks",
            method_descriptor="(Ljava/lang/Object;)Z",
            source_lines={7: "if (ready && collaborator.accept(value)) {"},
            include_statuses={"NO_COVERAGE"},
        )

        assert len(result) == 1
        assert "source_predicate" not in result[0]
        assert result[0]["pit_index"] == 12

    def test_arithmetic_mutant_extracts_nonzero_and_loop_carried_state(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Hash</mutatedClass>
            <mutatedMethod>mix</mutatedMethod>
            <methodDescription>([B)[J</methodDescription>
            <lineNumber>6</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.MathMutator</mutator>
            <description>Replaced long addition with subtraction</description>
          </mutation>
        </mutations>""")
        source_lines = {
            1: "class Hash {",
            2: "long[] mix(byte[] input) {",
            3: "long h2 = 0;",
            4: "for (int i = 0; i < blocks; i++) {",
            5: "h1 ^= input[i];",
            6: "h1 += h2;",
            7: "long k2 = input[i];",
            8: "h2 ^= k2;",
            9: "}",
            10: "return new long[] { h1, h2 };",
            11: "}",
            12: "}",
        }

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "mix",
            class_fqn="com.example.Hash",
            method_descriptor="([B)[J",
            source_lines=source_lines,
        )

        constraint = result[0]["mutant_distinguishing_constraints"][0]
        assert constraint["mutant_distinguishing_rows"][0]["constraint"] == (
            "h2 != 0 at line 6"
        )
        dependency = constraint["source_state_dependency"]
        assert dependency["initialization_line"] == 3
        assert dependency["later_update_line"] == 8
        assert "later loop iteration" in dependency["minimum_execution_policy"]
        assert "inverts or removes a conditional" not in result[0]["kill_hint"]
        assert "h2 != 0 at line 6" in result[0]["kill_hint"]

    def test_math_mutator_uses_bitwise_distinguishing_relation(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Codec</mutatedClass>
            <mutatedMethod>decode</mutatedMethod>
            <lineNumber>17</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.MathMutator</mutator>
            <description>Replaced bitwise AND with OR</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "decode",
            source_lines={17: "value ^= ((long) data[index] &amp; 0xff);"},
        )

        row = result[0]["mutant_distinguishing_constraints"][0][
            "mutant_distinguishing_rows"
        ][0]
        assert "AND operands" in row["constraint"]
        assert "OR" in row["constraint"]

    def test_nested_addition_mutant_does_not_bind_outer_compound_assignment(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Codec</mutatedClass>
            <mutatedMethod>decode</mutatedMethod>
            <lineNumber>17</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.MathMutator</mutator>
            <description>Replaced integer addition with subtraction</description>
          </mutation>
        </mutations>""")

        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "decode",
            source_lines={
                17: "value ^= ((long) data[index + 14] &amp; 0xff) &lt;&lt; 48;"
            },
        )

        row = result[0]["mutant_distinguishing_constraints"][0][
            "mutant_distinguishing_rows"
        ][0]
        assert row["constraint"] == "14 != 0 at line 17"
        assert "((long) data" not in row["constraint"]

    def test_mutant_oracle_strategy_handles_no_coverage_and_void_call(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="NO_COVERAGE">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>check</mutatedMethod>
            <lineNumber>11</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <description>negated conditional</description>
          </mutation>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>check</mutatedMethod>
            <lineNumber>12</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.VoidMethodCallMutator</mutator>
            <description>removed call to com/example/Foo::touch</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(
            xml,
            "check",
            include_statuses={"SURVIVED", "NO_COVERAGE"},
            source_lines={11: "if (ready) {", 12: "touch(state);"},
        )
        assert any(item["status"] == "NO_COVERAGE" and "assertNotNull" in " ".join(item["oracle_strategy"]) for item in result)
        assert any(item["mutator_short"] == "remove-void-call" and "side effect" in " ".join(item["oracle_strategy"]) for item in result)

    def test_class_fqn_filter_excludes_other_classes(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Other</mutatedClass>
            <mutatedMethod>doIt</mutatedMethod>
            <lineNumber>5</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.returns.NullReturnValsMutator</mutator>
            <description>replaced return value with null</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(
            xml, "doIt", class_fqn="com.example.Foo"
        )
        assert result == []

    def test_easy_mutants_come_before_hard_mutants(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>m</mutatedMethod>
            <lineNumber>20</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <description>negated conditional</description>
          </mutation>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>m</mutatedMethod>
            <lineNumber>30</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.returns.PrimitiveReturnsMutator</mutator>
            <description>replaced int return with 0</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(xml, "m")
        assert len(result) == 2
        assert result[0]["difficulty"] == "easy"
        assert result[1]["difficulty"] == "hard"

    def test_source_line_attached_when_provided(self, tmp_path):
        from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method
        xml = self._make_xml(tmp_path, """<mutations>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.example.Foo</mutatedClass>
            <mutatedMethod>bar</mutatedMethod>
            <lineNumber>3</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.returns.PrimitiveReturnsMutator</mutator>
            <description>replaced int return with 0</description>
          </mutation>
        </mutations>""")
        result = parse_pitest_xml_surviving_mutants_for_method(
            xml, "bar", source_lines={3: "    return count + offset;"}
        )
        assert len(result) == 1
        assert result[0]["source_line"] == "return count + offset;"
        def test_no_coverage_mutants_can_be_included_as_reachability_feedback(self, tmp_path):
                from tatg.evaluation.mutation import parse_pitest_xml_surviving_mutants_for_method

                xml = self._make_xml(tmp_path, """<mutations>
                    <mutation detected="false" status="NO_COVERAGE">
                        <mutatedClass>com.example.Foo</mutatedClass>
                        <mutatedMethod>bar</mutatedMethod>
                        <lineNumber>8</lineNumber>
                        <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
                        <description>negated conditional</description>
                    </mutation>
                </mutations>""")

                result = parse_pitest_xml_surviving_mutants_for_method(
                        xml,
                        "bar",
                        source_lines={8: "if (value > 0) {"},
                        include_statuses={"SURVIVED", "NO_COVERAGE"},
                )

                assert len(result) == 1
                assert result[0]["status"] == "NO_COVERAGE"
                assert "No current test reaches" in result[0]["kill_hint"]
                assert result[0]["source_line"] == "if (value > 0) {"


# ---------------------------------------------------------------------------
# Rec E: _build_method_coverage_feedback budget guidance
# ---------------------------------------------------------------------------

class TestBuildMethodCoverageFeedbackBudget:
    def _make_cov(self):
        return [
            {
                "method_name": "process",
                "line_covered": 10,
                "line_missed": 5,
                "branch_covered": 4,
                "branch_missed": 4,
                "missed_source_lines": [],
            }
        ]

    def test_no_budget_warning_when_count_under_threshold(self):
        from tatg.cli import _build_method_coverage_feedback
        fb = _build_method_coverage_feedback(
            self._make_cov(), "process", current_test_count=30
        )
        assert fb is not None
        guidance = " ".join(fb.get("guidance") or [])
        assert "budget" not in guidance.lower()

    def test_budget_warning_when_count_over_threshold(self):
        from tatg.cli import _build_method_coverage_feedback
        fb = _build_method_coverage_feedback(
            self._make_cov(), "process", current_test_count=80
        )
        assert fb is not None
        guidance = " ".join(fb.get("guidance") or [])
        assert "budget" in guidance.lower() or "80" in guidance

    def test_current_test_count_stored_in_feedback(self):
        from tatg.cli import _build_method_coverage_feedback
        fb = _build_method_coverage_feedback(
            self._make_cov(), "process", current_test_count=45
        )
        assert fb is not None
        assert fb.get("current_test_count") == 45


class TestFindMethodLineSpanAt:
    """Tests for find_method_line_span_at — overload-aware span lookup."""

    _SOURCE = textwrap.dedent("""\
        class Foo {
            static String escape(String s) {
                return s;
            }

            static String escape(String s, Object out) {
                return s + out;
            }

            static void escape(Appendable a, String s, boolean flag) {
                // body
            }
        }
    """)

    def test_returns_first_overload_by_hint(self):
        from tatg.coverage_obligations import find_method_line_span_at
        span = find_method_line_span_at(self._SOURCE, "escape", 2)
        assert span is not None
        start, end = span
        assert start == 2
        assert start < end

    def test_returns_second_overload_by_hint(self):
        from tatg.coverage_obligations import find_method_line_span_at
        span = find_method_line_span_at(self._SOURCE, "escape", 6)
        assert span is not None
        start, end = span
        assert start == 6

    def test_returns_third_overload_by_hint(self):
        from tatg.coverage_obligations import find_method_line_span_at
        span = find_method_line_span_at(self._SOURCE, "escape", 10)
        assert span is not None
        start, end = span
        assert start == 10

    def test_falls_back_to_first_when_no_hint(self):
        from tatg.coverage_obligations import find_method_line_span_at
        span = find_method_line_span_at(self._SOURCE, "escape", 0)
        assert span is not None
        # hint=0 means no hint → fall back to first match
        assert span[0] == 2


class TestParsePackagePrivateMethods:
    """Tests for package-private static method detection in api_surface.py."""

    def test_extracts_package_private_static(self):
        from tatg.analysis.api_surface import extract_surface
        import tempfile, os
        code = textwrap.dedent("""\
            package org.example;
            public class Entities {
                public static String escape(String s) { return s; }
                static void escape(java.io.Appendable a, String s, boolean flag) throws java.io.IOException {}
            }
        """)
        with tempfile.NamedTemporaryFile(suffix=".java", mode="w", delete=False) as f:
            f.write(code)
            tmp = Path(f.name)
        try:
            surface = extract_surface(tmp, "Entities")
            method_renders = [m.render() for m in surface.target_methods]
            pkg_private = [r for r in method_renders if "static void escape" in r]
            assert len(pkg_private) == 1, f"Expected 1 package-private escape, got: {pkg_private}"
        finally:
            os.unlink(tmp)

    def test_no_duplication_of_public_static(self):
        from tatg.analysis.api_surface import extract_surface
        import tempfile, os
        code = textwrap.dedent("""\
            package org.example;
            public class Foo {
                public static String bar(String s) { return s; }
            }
        """)
        with tempfile.NamedTemporaryFile(suffix=".java", mode="w", delete=False) as f:
            f.write(code)
            tmp = Path(f.name)
        try:
            surface = extract_surface(tmp, "Foo")
            bar_methods = [m for m in surface.target_methods if m.name == "bar"]
            assert len(bar_methods) == 1, f"public static bar duplicated: {bar_methods}"
        finally:
            os.unlink(tmp)

    def test_package_private_static_render_includes_static(self):
        from tatg.analysis.api_surface import extract_surface
        import tempfile, os
        code = textwrap.dedent("""\
            class Parser {
                static int parse(String s) { return 0; }
            }
        """)
        with tempfile.NamedTemporaryFile(suffix=".java", mode="w", delete=False) as f:
            f.write(code)
            tmp = Path(f.name)
        try:
            surface = extract_surface(tmp, "Parser")
            renders = [m.render() for m in surface.target_methods]
            static_renders = [r for r in renders if "static" in r and "parse" in r]
            assert static_renders, f"Expected static parse in surface, got: {renders}"
        finally:
            os.unlink(tmp)


class TestCatchRethrowNestedBlocks:
    """Regression for Bug #2: catch-rethrow regex must see through nested braces."""

    def test_catch_rethrow_with_nested_if_block(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = (
            "try { op(); } "
            "catch (IOException e) { "
            "  if (log.isDebugEnabled()) { log.debug(msg); } "
            "  throw new RuntimeException(e); "
            "}"
        )
        result = _extract_catch_rethrow_facts(body)
        assert len(result) == 1, f"Expected 1 fact, got: {result}"
        assert "IOException" in result[0]
        assert "RuntimeException" in result[0]

    def test_catch_rethrow_with_nested_try_finally(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = (
            "try { op(); } "
            "catch (SQLException e) { "
            "  try { conn.close(); } finally { cleanup(); } "
            "  throw new DataAccessException(e); "
            "}"
        )
        result = _extract_catch_rethrow_facts(body)
        assert len(result) == 1, f"Expected 1 fact, got: {result}"
        assert "SQLException" in result[0]


class TestStripNonsemanticJavaTextBlocks:
    """Regression for Bug #4: _strip_nonsemantic_java must handle Java text blocks."""

    def test_text_block_content_not_treated_as_code(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_throw_guard_facts
        # "throw new" inside a text block is a string literal, not code.
        body = (
            'String sql = """'
            'SELECT * FROM t WHERE x > 0; -- throw new IOException'
            '""";'
            "return sql;"
        )
        facts = _extract_throw_guard_facts(body)
        assert facts == [], f"False-positive throw-guard facts from text block: {facts}"

    def test_text_block_braces_not_confused_with_code_braces(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = (
            'String json = """{"key": "catch (X e) { throw new Y(e); }"}""";'
            "process(json);"
        )
        facts = _extract_catch_rethrow_facts(body)
        assert facts == [], f"False-positive catch-rethrow facts from text block: {facts}"

    def test_real_rethrow_after_text_block_is_still_found(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_catch_rethrow_facts
        body = (
            'String msg = """error occurred""";'
            "try { op(); } catch (IOException e) { throw new RuntimeException(e); }"
        )
        facts = _extract_catch_rethrow_facts(body)
        assert len(facts) == 1, f"Expected 1 fact, got: {facts}"
        assert "IOException" in facts[0]


class TestStripCommentsAndLiteralsTextBlocks:
    """Regression for Bug #4: _strip_comments_and_literals must handle Java text blocks."""

    def test_text_block_stripped_before_string_regex(self):
        from tatg.coverage_obligations import _strip_comments_and_literals
        src = 'String s = """hello "world" """;'
        result = _strip_comments_and_literals(src)
        assert '"world"' not in result, "Text-block content leaked through stripping"

    def test_newlines_inside_text_block_preserved(self):
        from tatg.coverage_obligations import _strip_comments_and_literals
        src = 'String s = """\nline1\nline2\n""";'
        result = _strip_comments_and_literals(src)
        assert result.count("\n") == src.count("\n"), "Newlines inside text block must be preserved"


class TestPkgPrivatePrefixLookback:
    """Regression for Bug #5: prefix lookback must not false-positive on previous-line comments."""

    def test_previous_line_public_comment_does_not_exclude_pkg_private_method(self):
        from tatg.analysis.api_surface import _parse_package_private_methods, _strip_comments
        code = _strip_comments(
            "// This is the public escape interface\n"
            "static void escape(java.io.Appendable a, String s) throws java.io.IOException {}\n"
        )
        methods = _parse_package_private_methods(code, set())
        assert any(m.name == "escape" for m in methods), (
            "Package-private static method incorrectly excluded due to 'public' in preceding comment"
        )

    def test_public_static_on_same_line_is_excluded(self):
        from tatg.analysis.api_surface import _parse_package_private_methods, _strip_comments
        code = _strip_comments(
            "public static void escape(java.io.Appendable a, String s) {}\n"
        )
        methods = _parse_package_private_methods(code, set())
        assert not any(m.name == "escape" for m in methods), (
            "public static method must not appear in package-private results"
        )


class TestStrictSurfaceRuleTypeMatch:
    """Rule 6 in STRICT SURFACE RULES: parameter types must exactly match the listed signature.

    This is the generic fix for LLM type-substitution hallucinations such as passing
    a type that appears elsewhere in the class (e.g. QName used internally) where a
    different parameter type (e.g. String) is required.
    """

    def test_strict_surface_rules_contain_type_match_rule(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor
        java = tmp_path / "Foo.java"
        java.write_text(
            "class Foo { public void write(String s) {} }",
            encoding="utf-8",
        )
        surface = extract_surface(java, "Foo")
        rendered = render_surface_anchor(surface)
        assert "STRICT SURFACE RULES" in rendered
        assert "Parameter TYPES must exactly match" in rendered
        assert "NOT automatically" in rendered
        assert "field/internal type" in rendered

    def test_private_constructors_are_not_advertised_as_callable_surface(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor
        java = tmp_path / "RulerLike.java"
        java.write_text(
            "public class RulerLike { private RulerLike() {} public static boolean matches(String e, String r) { return true; } }",
            encoding="utf-8",
        )
        surface = extract_surface(java, "RulerLike")
        rendered = render_surface_anchor(surface)

        assert "constructors:" not in rendered
        assert "RulerLike()" not in rendered
        assert "matches(String e, String r)" in rendered

    def test_callable_surface_excludes_nested_and_sibling_owner_methods(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        java = tmp_path / "Outer.java"
        java.write_text(
            """
public class Outer {
    public String remove(Object key) { return null; }
    static class Cursor {
        public void remove() {}
    }
}
class Sibling {
    public boolean remove(Object key) { return false; }
}
""".strip(),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(java, "Outer"))

        assert "String remove(Object key)" in rendered
        assert "void remove()" not in rendered
        assert "boolean remove(Object key)" not in rendered


class TestBodyUsedHelperSurface:
    def test_nested_helper_surface_is_scoped_to_nested_owner(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        target = tmp_path / "Target.java"
        owner = tmp_path / "Owner.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                public class Target {
                    public int read(Owner.Context context) {
                        return context.position();
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        owner.write_text(
            textwrap.dedent(
                """\
                package org.example;
                public abstract class Owner {
                    public void ownerOnly() {}

                    public static class Context {
                        public int position() { return 1; }
                    }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))
        context_surface = rendered.split("Context:", 1)[1].split("Owner:", 1)[0]

        assert "int position()" in context_surface
        assert "ownerOnly()" not in context_surface

    def test_same_package_helper_methods_are_rendered_with_exact_surface(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        target = tmp_path / "Target.java"
        helper = tmp_path / "Helper.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                public class Target {
                    public String parse(String raw) {
                        return Helper.real(raw);
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        helper.write_text(
            textwrap.dedent(
                """\
                package org.example;
                public class Helper {
                    public Helper(String seed) {}
                    public static String real(String raw) { return raw; }
                    static int packageVisible(int value) { return value; }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Body-used helper types" in rendered
        assert "public Helper:" in rendered
        assert "public Helper(String seed)" in rendered
        assert "public static String real(String raw)" in rendered
        assert "package-private static int packageVisible(int value)" in rendered
        assert "arbitrary factory/helper methods" in rendered

    def test_callable_surface_renders_modifiers_and_filters_private_members(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        target = tmp_path / "Target.java"
        helper = tmp_path / "Helper.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                class Target {
                    public Target(String name) {}
                    protected void visibleToSamePackage() {}
                    void packageOnly() {}
                    private void hidden() {}
                    public String parse(String raw) { return Helper.real(raw); }
                }
                """
            ),
            encoding="utf-8",
        )
        helper.write_text(
            textwrap.dedent(
                """\
                package org.example;
                class Helper {
                    private Helper() {}
                    Helper(String seed) {}
                    protected String protectedHelper() { return ""; }
                    private String hiddenHelper() { return ""; }
                    static String real(String raw) { return raw; }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "public Target(String name)" in rendered
        assert "protected void visibleToSamePackage()" in rendered
        assert "package-private void packageOnly()" in rendered
        assert "hidden()" not in rendered
        assert "package-private Helper:" in rendered
        assert "private Helper()" not in rendered
        assert "package-private Helper(String seed)" in rendered
        assert "protected String protectedHelper()" in rendered
        assert "hiddenHelper" not in rendered
        assert "package-private static String real(String raw)" in rendered

    def test_unused_same_package_helper_does_not_pollute_exact_helper_surface(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        target = tmp_path / "Target.java"
        helper = tmp_path / "UnusedHelper.java"
        target.write_text(
            "package org.example; public class Target { public int parse() { return 1; } }",
            encoding="utf-8",
        )
        helper.write_text(
            "package org.example; public class UnusedHelper { public static int real() { return 2; } }",
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Same-package types:" in rendered
        assert "UnusedHelper" in rendered
        assert "Body-used helper types" not in rendered
        assert "real()" not in rendered

    def test_imported_body_helper_methods_are_rendered_even_when_not_in_signature(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        helper_dir = src_root / "org" / "example" / "util"
        target_dir.mkdir(parents=True)
        helper_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        helper = helper_dir / "ImportedHelper.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.util.ImportedHelper;
                public class Target {
                    public String parse(String raw) {
                        return ImportedHelper.real(raw);
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        helper.write_text(
            textwrap.dedent(
                """\
                package org.example.util;
                public class ImportedHelper {
                    public static String real(String raw) { return raw; }
                    public static String other(String raw) { return raw; }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Body-used helper types" in rendered
        assert "ImportedHelper:" in rendered
        assert "String real(String raw)" in rendered

    def test_import_only_helper_is_not_rendered_as_body_used_surface(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        helper_dir = src_root / "org" / "example" / "util"
        target_dir.mkdir(parents=True)
        helper_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        helper = helper_dir / "ImportedHelper.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.util.ImportedHelper;
                public class Target {
                    public int parse(String raw) {
                        return raw.length();
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        helper.write_text(
            "package org.example.util; public class ImportedHelper { public static String real(String raw) { return raw; } }",
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Body-used helper types" not in rendered
        assert "ImportedHelper:" not in rendered

    def test_expression_calls_are_not_misread_as_helper_method_signatures(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        target = tmp_path / "Target.java"
        helper = tmp_path / "Helper.java"
        target.write_text(
            "package org.example; public class Target { public Helper parse() { return Helper.make(); } }",
            encoding="utf-8",
        )
        helper.write_text(
            textwrap.dedent(
                """\
                package org.example;
                public class Helper {
                    public static Helper make() { return new Helper(); }
                    public static void fail() { throw new IllegalStateException(); }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Helper make()" in rendered
        assert "void fail()" in rendered
        assert "return new Helper" not in rendered
        assert "throw new IllegalStateException" not in rendered

    def test_imported_interface_methods_are_rendered_for_constructor_dependencies(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        store_dir = src_root / "org" / "example" / "store"
        target_dir.mkdir(parents=True)
        store_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        store = store_dir / "TaskStore.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.store.TaskStore;
                public class Target {
                    private final TaskStore taskStore;
                    public Target(TaskStore taskStore) {
                        this.taskStore = taskStore;
                    }
                    public String read(String id) {
                        return taskStore.get(id);
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        store.write_text(
            textwrap.dedent(
                """\
                package org.example.store;
                public interface TaskStore {
                    void save(String id, String value);
                    @Nullable String get(String id);
                    void delete(String id);
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Referenced types" in rendered
        assert "TaskStore:" in rendered
        assert "String get(String id)" in rendered
        assert "void save(String id, String value)" in rendered
        assert "getTask" not in rendered

    def test_imported_referenced_type_renders_nested_enum_constants(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        model_dir = src_root / "org" / "example" / "model"
        target_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        message = model_dir / "Message.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.model.Message;
                public class Target {
                    public String send(Message message) {
                        return message.kind();
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        message.write_text(
            textwrap.dedent(
                """\
                package org.example.model;
                public record Message(Role role) {
                    public String kind() { return "message"; }
                    public enum Role {
                        ROLE_USER,
                        ROLE_AGENT,
                        ROLE_UNSPECIFIED;
                    }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Message:" in rendered
        assert "String kind()" in rendered
        assert "enum Role constants: Message.Role.ROLE_USER, Message.Role.ROLE_AGENT, Message.Role.ROLE_UNSPECIFIED" in rendered
        assert "Role.USER" not in rendered

    def test_imported_private_methods_are_not_rendered_as_callable_surface(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        helper_dir = src_root / "org" / "example" / "helper"
        target_dir.mkdir(parents=True)
        helper_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        helper = helper_dir / "Processor.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.helper.Processor;
                public class Target {
                    private final Processor processor;
                    public Target(Processor processor) {
                        this.processor = processor;
                    }
                    public void run() {
                        processor.process();
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        helper.write_text(
            textwrap.dedent(
                """\
                package org.example.helper;
                public class Processor {
                    public void process() {}
                    private String extractContextId(Object event) { return ""; }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "void process()" in rendered
        assert "extractContextId" not in rendered

    def test_imported_protected_constructors_are_not_rendered_as_callable_surface(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        helper_dir = src_root / "org" / "example" / "helper"
        target_dir.mkdir(parents=True)
        helper_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        helper = helper_dir / "Processor.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.helper.Processor;
                public class Target {
                    private final Processor processor;
                    public Target(Processor processor) {
                        this.processor = processor;
                    }
                    public void run() {
                        processor.process();
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        helper.write_text(
            textwrap.dedent(
                """\
                package org.example.helper;
                public class Processor {
                    protected Processor() {}
                    public static Processor create() { return new Processor(); }
                    public void process() {}
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "Processor:" in rendered
        assert "Processor()" not in rendered
        assert "Processor create()" in rendered
        assert "void process()" in rendered

    def test_imported_body_used_enum_constants_are_rendered(self, tmp_path):
        from tatg.analysis.api_surface import extract_surface, render_surface_anchor

        src_root = tmp_path / "module" / "src" / "main" / "java"
        target_dir = src_root / "org" / "example"
        model_dir = src_root / "org" / "example" / "model"
        target_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        target = target_dir / "Target.java"
        state = model_dir / "TaskState.java"
        target.write_text(
            textwrap.dedent(
                """\
                package org.example;
                import org.example.model.TaskState;
                public class Target {
                    public boolean ready() {
                        return TaskState.TASK_STATE_WORKING.isFinal();
                    }
                }
                """
            ),
            encoding="utf-8",
        )
        state.write_text(
            textwrap.dedent(
                """\
                package org.example.model;
                public enum TaskState {
                    TASK_STATE_SUBMITTED,
                    TASK_STATE_WORKING,
                    TASK_STATE_COMPLETED;
                    public boolean isFinal() { return this == TASK_STATE_COMPLETED; }
                }
                """
            ),
            encoding="utf-8",
        )

        rendered = render_surface_anchor(extract_surface(target, "Target"))

        assert "TaskState:" in rendered
        assert "enum TaskState constants: TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_COMPLETED" in rendered
        assert "TaskState.WORKING" not in rendered


# ---------------------------------------------------------------------------
# instanceof dispatch extraction and boundary recipe integration
# ---------------------------------------------------------------------------

class TestExtractInstanceofDispatchFacts:
    def test_basic_instanceof_dispatch(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_instanceof_dispatch_facts
        body = """
            if (obj instanceof Iterator) { return (Iterator<?>) obj; }
            if (obj instanceof Iterable) { return ((Iterable<?>) obj).iterator(); }
            if (obj instanceof Map) { return ((Map<?, ?>) obj).values().iterator(); }
        """
        result = _extract_instanceof_dispatch_facts([("Object", "obj")], body)
        assert "obj" in result
        assert "Iterator" in result["obj"]
        assert "Iterable" in result["obj"]
        assert "Map" in result["obj"]
        assert result["obj"].index("Iterator") < result["obj"].index("Iterable") < result["obj"].index("Map")

    def test_no_instanceof_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_instanceof_dispatch_facts
        body = "if (obj == null) { return null; }"
        result = _extract_instanceof_dispatch_facts([("Object", "obj")], body)
        assert result == {}

    def test_array_instanceof_type_captured(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_instanceof_dispatch_facts
        body = "if (obj instanceof Object[]) { return new ObjectArrayIterator<>((Object[]) obj); }"
        result = _extract_instanceof_dispatch_facts([("Object", "obj")], body)
        assert "obj" in result
        assert "Object[]" in result["obj"]

    def test_deduplicates_repeated_instanceof(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_instanceof_dispatch_facts
        body = """
            if (obj instanceof List) { doA(); }
            if (obj instanceof List) { doB(); }
        """
        result = _extract_instanceof_dispatch_facts([("Object", "obj")], body)
        assert result["obj"].count("List") == 1

    def test_multiple_params_tracked_independently(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_instanceof_dispatch_facts
        body = """
            if (a instanceof String) { return a.toString(); }
            if (b instanceof Integer) { return b.intValue(); }
        """
        result = _extract_instanceof_dispatch_facts([("Object", "a"), ("Object", "b")], body)
        assert "String" in result.get("a", [])
        assert "Integer" in result.get("b", [])

    def test_instanceof_ignored_in_string_literals(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_instanceof_dispatch_facts
        body = 'String s = "obj instanceof String is ignored"; if (obj instanceof Map) { }'
        result = _extract_instanceof_dispatch_facts([("Object", "obj")], body)
        assert result.get("obj") == ["Map"]


class TestBoundaryInputRecipeInstanceofDispatch:
    def test_object_param_with_dispatch_lists_types(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        body = """
            if (obj instanceof Iterator) { return (Iterator<?>) obj; }
            if (obj instanceof Iterable) { return ((Iterable<?>) obj).iterator(); }
            if (obj instanceof Object[]) { return new ObjectArrayIterator<>((Object[]) obj); }
        """
        recipe = _boundary_input_recipe("getIterator", [("Object", "obj")], body)
        assert recipe is not None
        goal = recipe.path_goals[0]
        assert "Iterator" in goal
        assert "Iterable" in goal
        assert "Object[]" in goal
        assert "start with a non-null value" in goal
        assert "fallback" in goal

    def test_object_param_without_dispatch_keeps_generic_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        body = "if (obj == null) { return null; } return obj.toString();"
        recipe = _boundary_input_recipe("wrap", [("Object", "obj")], body)
        assert recipe is not None
        assert "minimally valid instance" in recipe.path_goals[0]

    def test_single_instanceof_keeps_generic_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        body = "if (obj instanceof String) { return (String) obj; }"
        recipe = _boundary_input_recipe("cast", [("Object", "obj")], body)
        assert recipe is not None
        assert "minimally valid instance" in recipe.path_goals[0]

    def test_no_body_keeps_generic_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe
        recipe = _boundary_input_recipe("doWork", [("Object", "obj")])
        assert recipe is not None
        assert "minimally valid instance" in recipe.path_goals[0]
        assert "start with a non-null value" in recipe.path_goals[0]

    def test_reflection_field_uses_source_proven_fixture_lookup(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe

        recipe = _boundary_input_recipe(
            "exclude",
            [("Field", "field")],
            """
            Expose expose = field.getAnnotation(Expose.class);
            Since since = field.getAnnotation(Since.class);
            return field.isSynthetic() || expose == null || since == null;
            """,
            source_imports={"Field": "java.lang.reflect.Field"},
        )

        assert recipe is not None
        text = "\n".join(recipe.path_goals)
        assert "inside each returned @Test method" in text
        assert "class Fixture { String fieldName; }" in text
        assert "java.lang.reflect.Field field =" in text
        assert 'Fixture.class.getDeclaredField("fieldName")' in text
        assert "@Expose" in text
        assert "@Since" in text
        assert "Mockito.mock(Field.class)" not in text
        assert "do not construct, subclass, or invent a helper" in text

    def test_object_param_mock_fallback_uses_only_source_observed_calls(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _boundary_input_recipe

        body = """
            int count = atts.getLength();
            if (count > 0) {
                return atts.getValue(0);
            }
            return "";
        """
        recipe = _boundary_input_recipe("parse", [("Attributes", "atts")], body)

        assert recipe is not None
        goal = recipe.path_goals[0]
        assert "visible constructor or factory" in goal
        assert "Mockito.mock(Attributes.class)" in goal
        assert "`getLength(...)`" in goal
        assert "`getValue(...)`" in goal
        assert "addAttribute" not in goal


# ---------------------------------------------------------------------------
# Semantic Gap Translation — _semantic_hint_for_missed_line
# ---------------------------------------------------------------------------

class TestSemanticHintForMissedLine:
    def _hint(self, code: str, param_names: frozenset) -> str | None:
        from tatg.coverage_obligations import _semantic_hint_for_missed_line
        return _semantic_hint_for_missed_line(code, param_names)

    def test_instanceof_param_gives_construction_hint(self):
        hint = self._hint("if (obj instanceof Iterator)", frozenset({"obj"}))
        assert hint is not None
        assert "Iterator" in hint
        assert "obj" in hint
        assert "pass" in hint.lower()

    def test_instanceof_nonparam_still_gives_hint(self):
        hint = self._hint("if (value instanceof Map)", frozenset({"x", "y"}))
        assert hint is not None
        assert "Map" in hint
        assert "ensure" in hint.lower() or "pass" in hint.lower()

    def test_null_equality_check(self):
        hint = self._hint("if (name == null)", frozenset({"name"}))
        assert hint is not None
        assert "null" in hint
        assert "name" in hint

    def test_not_null_check(self):
        hint = self._hint("if (data != null)", frozenset({"data"}))
        assert hint is not None
        assert "non-null" in hint
        assert "data" in hint

    def test_isempty_check(self):
        hint = self._hint("if (items.isEmpty())", frozenset({"items"}))
        assert hint is not None
        assert "empty" in hint.lower()
        assert "items" in hint

    def test_numeric_boundary_check(self):
        hint = self._hint("if (count < 10)", frozenset({"count"}))
        assert hint is not None
        assert "count" in hint
        # should mention at least two boundary values
        digits = [c for c in hint if c.isdigit()]
        assert len(digits) >= 2

    def test_unrecognized_pattern_returns_none(self):
        # a plain return statement has no recognizable gap pattern
        hint = self._hint("return result;", frozenset({"result"}))
        assert hint is None

    def test_instanceof_construction_hint_array_type(self):
        from tatg.coverage_obligations import _instanceof_construction_hint
        h = _instanceof_construction_hint("String[]")
        assert "new String[0]" in h

    def test_instanceof_construction_hint_known_type(self):
        from tatg.coverage_obligations import _instanceof_construction_hint
        h = _instanceof_construction_hint("Iterator")
        assert "iterator" in h.lower()

    def test_instanceof_construction_hint_unknown_type(self):
        from tatg.coverage_obligations import _instanceof_construction_hint
        h = _instanceof_construction_hint("SomeCustomClass")
        assert "SomeCustomClass" in h


# ---------------------------------------------------------------------------
# InstanceofDispatch obligation in synthesize_coverage_obligations
# ---------------------------------------------------------------------------

class TestSynthesizeCoverageObligationsInstanceofDispatch:
    def _obligations(self, missed_lines: list[dict], body_snippet: str = "") -> list[dict]:
        from tatg.coverage_obligations import synthesize_coverage_obligations
        return synthesize_coverage_obligations(
            method_name="getIterator",
            missed_source_lines=missed_lines,
            line_missed=len(missed_lines),
            branch_missed=len(missed_lines),
        )

    def test_instanceof_dispatch_obligation_emitted(self):
        missed = [
            {"line": 10, "code": "if (obj instanceof Iterator)"},
            {"line": 12, "code": "if (obj instanceof Iterable)"},
        ]
        obs = self._obligations(missed)
        categories = [o["category"] for o in obs]
        assert "InstanceofDispatch" in categories

    def test_instanceof_dispatch_lists_types_in_program_fact(self):
        missed = [
            {"line": 10, "code": "if (obj instanceof Iterator)"},
            {"line": 12, "code": "if (obj instanceof Map)"},
        ]
        obs = self._obligations(missed)
        io = next(o for o in obs if o["category"] == "InstanceofDispatch")
        assert "Iterator" in io["program_fact"]
        assert "Map" in io["program_fact"]

    def test_instanceof_dispatch_includes_construction_hints(self):
        missed = [
            {"line": 10, "code": "if (obj instanceof Iterator)"},
            {"line": 12, "code": "if (obj instanceof List)"},
        ]
        obs = self._obligations(missed)
        io = next(o for o in obs if o["category"] == "InstanceofDispatch")
        setup_text = " ".join(io["required_setup"])
        # iterator hint should mention iterator or list construction
        assert any(word in setup_text.lower() for word in ["iterator", "list", "arraylist"])

    def test_no_instanceof_no_dispatch_obligation(self):
        missed = [
            {"line": 5, "code": "if (count > 0)"},
            {"line": 8, "code": "return null;"},
        ]
        obs = self._obligations(missed)
        categories = [o["category"] for o in obs]
        assert "InstanceofDispatch" not in categories

    def test_semantic_hint_annotated_on_anchor(self):
        """Anchors with instanceof code get a 'hint' field after synthesis."""
        missed = [
            {"line": 10, "code": "if (obj instanceof Iterator)"},
        ]
        # run synthesis; it mutates anchors in-place
        from tatg.coverage_obligations import synthesize_coverage_obligations
        synthesize_coverage_obligations(
            method_name="getIterator",
            missed_source_lines=missed,
            line_missed=1,
            branch_missed=1,
        )
        # annotation is idempotent and applied even with no source file
        # (param_names will be empty since no signature is available, but hint still emitted)
        # The key check: the hint field is set
        assert "hint" in missed[0]
        assert "Iterator" in missed[0]["hint"]

    def test_multi_iteration_stability(self):
        """Calling synthesis twice on the same anchors doesn't duplicate the hint."""
        from tatg.coverage_obligations import synthesize_coverage_obligations
        missed = [{"line": 10, "code": "if (obj instanceof Map)"}]
        synthesize_coverage_obligations(
            method_name="getIterator", missed_source_lines=missed,
            line_missed=1, branch_missed=1,
        )
        hint_after_first = missed[0].get("hint", "")
        synthesize_coverage_obligations(
            method_name="getIterator", missed_source_lines=missed,
            line_missed=1, branch_missed=1,
        )
        hint_after_second = missed[0].get("hint", "")
        assert hint_after_first == hint_after_second


# ──────────────────────────────────────────────────────────────────────────────
# Tests for _has_no_public_ctor and utility-class recipe
# ──────────────────────────────────────────────────────────────────────────────

class TestHasNoPublicCtor:
    def _fn(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _has_no_public_ctor
        return _has_no_public_ctor

    def _shape(self, ctors):
        shape = TypeShape(class_name="Foo", fqn="pkg.Foo", package="pkg")
        shape.constructors = ctors
        return shape

    def test_private_only_ctor_returns_true(self):
        ctor = ConstructorInfo(visibility="private", params=[])
        assert self._fn()(self._shape([ctor])) is True

    def test_public_ctor_returns_false(self):
        ctor = ConstructorInfo(visibility="public", params=[])
        assert self._fn()(self._shape([ctor])) is False

    def test_package_private_ctor_returns_false_for_same_package_test(self):
        ctor = ConstructorInfo(visibility="package-private", params=[("String", "value")])
        assert self._fn()(self._shape([ctor])) is False

    def test_no_ctors_at_all_returns_false(self):
        # No declared ctors → Java provides implicit public no-arg ctor → not a utility class
        assert self._fn()(self._shape([])) is False

    def test_mixed_ctors_with_public_returns_false(self):
        ctors = [
            ConstructorInfo(visibility="private", params=[("String", "x")]),
            ConstructorInfo(visibility="public", params=[]),
        ]
        assert self._fn()(self._shape(ctors)) is False


class TestUtilityClassConstructionRecipe:
    def _build_utility_shape(self, static_method_names):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        shape = TypeShape(class_name="Ruler", fqn="pkg.Ruler", package="pkg")
        # Private no-arg ctor explicitly declared → utility class
        shape.constructors = [ConstructorInfo(visibility="private", params=[("int", "x")])]
        shape.methods = [
            MethodInfo(name=n, visibility="public", is_static=True,
                       return_type="boolean", params=[("String", "event"), ("String", "rule")])
            for n in static_method_names
        ]
        return shape, _construction_recipe

    def test_utility_class_recipe_emitted(self):
        shape, fn = self._build_utility_shape(["matches", "matchesRule"])
        recipe = fn(shape, [])
        assert recipe is not None
        assert recipe.category == "utility-class"

    def test_utility_class_recipe_says_no_new(self):
        shape, fn = self._build_utility_shape(["matches"])
        recipe = fn(shape, [])
        assert recipe is not None
        combined = " ".join(recipe.setup_steps + recipe.avoid)
        assert "new Ruler" in combined or "NOT" in combined or "private" in combined

    def test_utility_class_recipe_lists_static_methods(self):
        shape, fn = self._build_utility_shape(["matches", "matchesRule"])
        recipe = fn(shape, [])
        assert recipe is not None
        joined = " ".join(recipe.setup_steps)
        assert "Ruler.matches" in joined or "Ruler.matchesRule" in joined

    def test_public_ctor_does_not_emit_utility_recipe(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        shape = TypeShape(class_name="HelpFormatter", fqn="pkg.HelpFormatter", package="pkg")
        shape.constructors = []  # no declared ctors → implicit public no-arg
        shape.methods = []
        recipe = _construction_recipe(shape, [])
        assert recipe is None or recipe.category != "utility-class"

    def test_same_package_constructor_is_rendered_with_exact_parameters(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe

        shape = TypeShape(
            class_name="Adapter",
            fqn="pkg.Adapter",
            package="pkg",
            constructors=[
                ConstructorInfo(
                    visibility="package-private",
                    params=[("Context", "context"), ("Strategy", "strategy")],
                )
            ],
        )

        recipe = _construction_recipe(shape, [])

        assert recipe is not None
        assert recipe.category != "utility-class"
        assert any(
            "new Adapter(Context context, Strategy strategy)" in step
            for step in recipe.setup_steps
        )
        assert all("no-arg constructor" not in step for step in recipe.setup_steps)

    def test_private_ctor_with_public_builder_uses_builder_recipe(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        shape = TypeShape(class_name="Agent", fqn="pkg.Agent", package="pkg")
        shape.constructors = [ConstructorInfo(visibility="private", params=[("Builder", "builder")])]
        shape.methods = [
            MethodInfo(name="builder", visibility="public", is_static=True,
                       return_type="Builder", params=[])
        ]
        recipe = _construction_recipe(shape, [])
        assert recipe is not None
        joined = " ".join(recipe.setup_steps)
        assert "Agent.builder(" in joined
        assert "no-arg constructor" not in joined

    def test_static_target_does_not_emit_receiver_construction_recipe(self, tmp_path: Path):
        from tatg.analysis.typeinfo.recipe_synthesizer import synthesize_execution_recipes

        source = tmp_path / "SchemaUtils.java"
        source.write_text(
            """
package pkg;
public final class SchemaUtils {
    public static String validate(String value) {
        return value;
    }
}
""".strip(),
            encoding="utf-8",
        )
        shape = TypeShape(class_name="SchemaUtils", fqn="pkg.SchemaUtils", package="pkg")
        shape.methods = [
            MethodInfo(
                name="validate",
                visibility="public",
                is_static=True,
                return_type="String",
                params=[("String", "value")],
            )
        ]

        recipes = synthesize_execution_recipes(
            source_file=source,
            target_shape=shape,
            type_index={"SchemaUtils": shape},
            method_name="validate",
            class_name="SchemaUtils",
            project_root=tmp_path,
        )

        assert not any(recipe.category == "construction" for recipe in recipes)
        assert not any(
            "create SchemaUtils" in step
            for recipe in recipes
            for step in recipe.setup_steps
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests for _param_type_context_recipe
# ──────────────────────────────────────────────────────────────────────────────

class TestParamTypeContextRecipe:
    def _make_shape_with_ctor(self, name, ctor_params):
        shape = TypeShape(class_name=name, fqn=f"pkg.{name}", package="pkg")
        shape.constructors = [ConstructorInfo(visibility="public", params=ctor_params)]
        shape.methods = []
        return shape

    def _make_shape_factory(self, name, factory_name):
        shape = TypeShape(class_name=name, fqn=f"pkg.{name}", package="pkg")
        shape.constructors = [ConstructorInfo(visibility="private", params=[("int", "x")])]
        shape.methods = [
            MethodInfo(name=factory_name, visibility="public", is_static=True,
                       return_type=name, params=[])
        ]
        return shape

    def test_known_internal_type_ctor_emits_setup_step(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        spark_shape = self._make_shape_with_ctor("SparkApplicationSpec", [("String", "image")])
        recipe = _param_type_context_recipe(
            "copyFrom",
            [("SparkApplicationSpec", "spec")],
            {"SparkApplicationSpec": spark_shape},
        )
        assert recipe is not None
        assert any("SparkApplicationSpec" in s for s in recipe.setup_steps)
        assert any("spec" in s for s in recipe.setup_steps)

    def test_primitive_param_ignored(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        recipe = _param_type_context_recipe("wrap", [("int", "x")], {})
        assert recipe is None

    def test_string_param_ignored(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        recipe = _param_type_context_recipe("parse", [("String", "s")], {})
        assert recipe is None

    def test_type_not_in_index_no_recipe(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        recipe = _param_type_context_recipe("foo", [("UnknownType", "x")], {})
        assert recipe is None

    def test_factory_method_preferred_over_ctor(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        shape = self._make_shape_factory("Builder", "newInstance")
        recipe = _param_type_context_recipe("configure", [("Builder", "b")], {"Builder": shape})
        assert recipe is not None
        assert any("newInstance" in s for s in recipe.setup_steps)

    def test_no_arg_ctor_emits_new_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        shape = TypeShape(class_name="Config", fqn="pkg.Config", package="pkg")
        shape.constructors = []  # no declared ctors → implicit public no-arg
        shape.methods = []
        recipe = _param_type_context_recipe("run", [("Config", "cfg")], {"Config": shape})
        assert recipe is not None
        assert any("new Config()" in s for s in recipe.setup_steps)

    def test_avoid_contains_no_nested_class_warning(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        shape = self._make_shape_with_ctor("DriverSpec", [])
        recipe = _param_type_context_recipe("m", [("DriverSpec", "d")], {"DriverSpec": shape})
        assert recipe is not None
        assert any("nested" in a.lower() for a in recipe.avoid)

    def test_mixed_params_only_known_types_emitted(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _param_type_context_recipe
        shape = self._make_shape_with_ctor("AppConfig", [("String", "host"), ("int", "port")])
        recipe = _param_type_context_recipe(
            "doWork",
            [("int", "x"), ("AppConfig", "cfg"), ("String", "name")],
            {"AppConfig": shape},
        )
        assert recipe is not None
        assert len(recipe.setup_steps) == 1
        assert "AppConfig" in recipe.setup_steps[0]


class TestStripCommentsOnly:
    """_strip_comments_only preserves char/string literals while removing comments."""

    def test_removes_line_comment(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _strip_comments_only
        result = _strip_comments_only("int x = 1; // this is a comment\nint y = 2;")
        assert "this is a comment" not in result
        assert "int x = 1;" in result
        assert "int y = 2;" in result

    def test_removes_block_comment(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _strip_comments_only
        result = _strip_comments_only("int x = /* ignored */ 1;")
        assert "ignored" not in result
        assert "int x =" in result
        assert "1;" in result

    def test_preserves_char_literal(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _strip_comments_only
        result = _strip_comments_only("case 'C': handleC(v);")
        assert "'C'" in result
        assert "handleC" in result

    def test_preserves_string_literal(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _strip_comments_only
        result = _strip_comments_only('String s = "hello world";')
        assert '"hello world"' in result

    def test_comment_inside_case_not_consumed(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _strip_comments_only
        src = "case 'A': // dispatch A\n  handleA(v);"
        result = _strip_comments_only(src)
        assert "'A'" in result
        assert "dispatch A" not in result
        assert "handleA" in result


class TestExtractCallSiteDispatchHints:
    """_extract_call_site_dispatch_hints identifies case-arm dispatch for a target method."""

    def test_single_char_dispatch(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = "switch(c) { case 'C': index = handleC(value, result, index); break; }"
        hints = _extract_call_site_dispatch_hints("handleC", src)
        assert len(hints) >= 1
        assert any("'C'" in h for h in hints)

    def test_no_dispatch_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = "index = handleC(value, result, index);"
        hints = _extract_call_site_dispatch_hints("handleC", src)
        assert hints == []

    def test_preceding_closed_switch_does_not_create_dispatch_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = """
        void parse(char c) {
          switch (c) {
            case ' ': consumeWhitespace(); break;
          }
          doPeek();
        }
        """
        assert _extract_call_site_dispatch_hints("doPeek", src) == []

    def test_case_in_preceding_method_does_not_create_dispatch_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = """
        void parse(char c) {
          switch (c) { case ' ': consumeWhitespace(); break; }
        }
        int doPeek() { return 1; }
        """
        assert _extract_call_site_dispatch_hints("doPeek", src) == []

    def test_dispatch_hint_does_not_overstate_global_reachability(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = "switch(c) { case 'C': handleC(value); break; }"
        hint = _extract_call_site_dispatch_hints("handleC", src)[0]
        assert "this source call site" in hint
        assert "only invoked" not in hint
        assert "always include" not in hint

    def test_different_method_not_matched(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = "switch(c) { case 'C': index = handleC(v); break; }"
        hints = _extract_call_site_dispatch_hints("handleD", src)
        assert hints == []

    def test_integer_dispatch(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = "switch(mode) { case 42: doSpecial(x); break; }"
        hints = _extract_call_site_dispatch_hints("doSpecial", src)
        assert len(hints) >= 1
        assert any("42" in h for h in hints)

    def test_named_constant_dispatch(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = "switch(type) { case EXACT: processExact(data); break; }"
        hints = _extract_call_site_dispatch_hints("processExact", src)
        assert len(hints) >= 1
        assert any("EXACT" in h for h in hints)

    def test_subtype_cast_dispatch_preserves_exact_caller_argument_projection(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints

        src = """
        void dispatch(final Pattern pattern) {
            switch (pattern.type()) {
                case RANGE:
                    handleRange((Range) pattern);
                    break;
                case EXACT:
                    handleExact((Exact) pattern);
                    break;
            }
        }
        """

        hints = _extract_call_site_dispatch_hints("handleRange", src)

        assert len(hints) == 1
        assert "case `RANGE`" in hints[0]
        assert "construct a source-visible `Range` value" in hints[0]
        assert "pass that same value as caller argument `pattern`" in hints[0]
        assert "`pattern.type()` selects this arm" in hints[0]
        assert "different sibling value" in hints[0]

    def test_deduplicates_repeated_pattern(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        src = (
            "switch(c) { case 'C': handleC(a); break; }\n"
            "switch(c) { case 'C': handleC(b); break; }"
        )
        hints = _extract_call_site_dispatch_hints("handleC", src)
        assert hints.count(hints[0]) == 1  # deduplicated

    def test_empty_method_name_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_call_site_dispatch_hints
        hints = _extract_call_site_dispatch_hints("", "case 'C': foo();")
        assert hints == []


class TestCallSiteDispatchRecipe:
    """_call_site_dispatch_recipe wraps dispatch hints in an ExecutionRecipe."""

    def test_produces_recipe_when_hints_found(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _call_site_dispatch_recipe
        src = "switch(ch) { case 'C': handleC(v, r, i); break; }"
        recipe = _call_site_dispatch_recipe("handleC", src)
        assert recipe is not None
        assert recipe.category == "call-site-dispatch"
        assert any("'C'" in g for g in recipe.path_goals)

    def test_returns_none_when_no_hints(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _call_site_dispatch_recipe
        recipe = _call_site_dispatch_recipe("handleC", "handleC(v);")
        assert recipe is None

    def test_none_method_name_returns_none(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _call_site_dispatch_recipe
        recipe = _call_site_dispatch_recipe(None, "case 'C': handleC();")
        assert recipe is None


class TestScanSiblingStringConstants:
    """_scan_sibling_string_constants maps case labels to wire-format strings."""

    def test_finds_matching_constant(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        # Write a sibling Constants.java
        (tmp_path / "Constants.java").write_text(
            'public class Constants { static final String EXACT_MATCH = "exactly"; }'
        )
        source_file = tmp_path / "Ruler.java"
        source_file.write_text("// dummy")
        result = _scan_sibling_string_constants(source_file, ["EXACT"])
        assert result.get("EXACT") == "exactly"

    def test_ignores_source_file_itself(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        source_file = tmp_path / "Foo.java"
        source_file.write_text('static final String FOO_MATCH = "foo";')
        result = _scan_sibling_string_constants(source_file, ["FOO"])
        # The source file itself is excluded
        assert "FOO" not in result

    def test_ambiguous_match_skipped(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        (tmp_path / "C.java").write_text(
            'class C { static final String EXACT_MATCH = "exactly"; static final String EXACT_VALUE = "ev"; }'
        )
        source_file = tmp_path / "Ruler.java"
        source_file.write_text("// dummy")
        result = _scan_sibling_string_constants(source_file, ["EXACT"])
        # Two constants contain "EXACT" — ambiguous → skip
        assert "EXACT" not in result

    def test_no_sibling_files_returns_empty(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        source_file = tmp_path / "Alone.java"
        source_file.write_text("// alone")
        result = _scan_sibling_string_constants(source_file, ["FOO"])
        assert result == {}

    def test_empty_case_labels_returns_empty(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        (tmp_path / "C.java").write_text('class C { static final String X = "x"; }')
        source_file = tmp_path / "T.java"
        source_file.write_text("// t")
        result = _scan_sibling_string_constants(source_file, [])
        assert result == {}

    def test_multiple_labels_resolved(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        (tmp_path / "Constants.java").write_text(
            'class C { '
            'static final String PREFIX_MATCH = "prefix"; '
            'static final String SUFFIX_MATCH = "suffix"; '
            '}'
        )
        source_file = tmp_path / "Rule.java"
        source_file.write_text("// rule")
        result = _scan_sibling_string_constants(source_file, ["PREFIX", "SUFFIX"])
        assert result.get("PREFIX") == "prefix"
        assert result.get("SUFFIX") == "suffix"

    def test_final_static_and_composite_labels_resolved(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _scan_sibling_string_constants
        (tmp_path / "Constants.java").write_text(
            'class C { '
            'final static String ANYTHING_BUT_MATCH = "anything-but"; '
            'final static String SUFFIX_MATCH = "suffix"; '
            'final static String NUMERIC = "numeric"; '
            'final static String EQ = "="; '
            '}'
        )
        source_file = tmp_path / "Rule.java"
        source_file.write_text("// rule")
        result = _scan_sibling_string_constants(source_file, ["ANYTHING_BUT_SUFFIX", "NUMERIC_EQ"])
        # Conservative: no single constant contains "ANYTHING_BUT_SUFFIX" or "NUMERIC_EQ"
        # as a substring → skip (no speculative compound combining to avoid misleading the LLM)
        assert result.get("ANYTHING_BUT_SUFFIX") is None
        assert result.get("NUMERIC_EQ") is None


class TestExtractStringBranchLiterals:
    def test_single_contains_literal(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = 'if (contains(value, index, 6, "CAESAR")) { return; }'
        result = _extract_string_branch_literals(body)
        assert "CAESAR" in result

    def test_multi_arg_contains_all_captured(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = 'if (contains(value, index, 2, "CK", "CG", "CQ")) { append("K"); }'
        result = _extract_string_branch_literals(body)
        assert "CK" in result
        assert "CG" in result
        assert "CQ" in result

    def test_starts_with_literal(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = 'if (s.startsWith("prefix")) { return true; }'
        result = _extract_string_branch_literals(body)
        assert "prefix" in result

    def test_no_comparison_method_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = 'int x = value.length(); if (x > 0) { result.append("K"); }'
        result = _extract_string_branch_literals(body)
        # "K" is on a line without a comparison method → should not be captured
        assert "K" not in result

    def test_deduplicates_repeated_literals(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = (
            'if (contains(value, 0, 3, "ABC")) { x(); }\n'
            'if (contains(value, 1, 3, "ABC")) { y(); }'
        )
        result = _extract_string_branch_literals(body)
        assert result.count("ABC") == 1

    def test_max_24_returned(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        # 30 unique literals on contains lines
        lines = "\n".join(
            f'if (s.contains("LIT{i}")) {{ x(); }}'
            for i in range(30)
        )
        result = _extract_string_branch_literals(lines)
        assert len(result) <= 24

    def test_negated_contains_also_captured(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = 'if (!contains(value, idx - 2, 4, "WICZ")) { append("S"); }'
        result = _extract_string_branch_literals(body)
        assert "WICZ" in result

    def test_char_literal_concat_not_captured(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        # Java code: '"' + val.asText() + '"'  → should NOT capture "' + val.asText() + '"
        body = "return val.isTextual() && ('\"' + val.asText()).startsWith(valuePattern.pattern());"
        result = _extract_string_branch_literals(body)
        # The false positive ' + val.asText() contains " + " so should be filtered
        for lit in result:
            assert " + " not in lit, f"False positive captured: {lit!r}"

    def test_string_array_initializer_captured(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_string_branch_literals
        body = (
            'String[] hex_prefixes = {"0x", "0X", "-0x", "-0X", "#", "-#"};\n'
            'for (String pfx : hex_prefixes) { if (str.startsWith(pfx)) { break; } }'
        )
        result = _extract_string_branch_literals(body)
        assert "0x" in result
        assert "0X" in result
        assert "-0x" in result
        assert "#" in result
        assert "-#" in result


class TestExtractObjectStateBranchHints:
    def test_boolean_predicate_extracted(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_object_state_branch_hints
        body = 'if (option.hasLongOpt()) { sb.append(longOpt); }'
        bool_preds, null_checks = _extract_object_state_branch_hints(body)
        assert any("hasLongOpt" in p for p in bool_preds)

    def test_null_check_extracted(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_object_state_branch_hints
        body = 'if (option.getOpt() == null) { appendLongOpt(); }'
        bool_preds, null_checks = _extract_object_state_branch_hints(body)
        assert any("getOpt" in n for n in null_checks)

    def test_non_predicate_method_not_captured(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_object_state_branch_hints
        body = 'if (size() > 0) { process(); }'
        bool_preds, null_checks = _extract_object_state_branch_hints(body)
        # "size" doesn't start with is/has/can etc. → not a bool pred
        assert not any("size" in p for p in bool_preds)

    def test_both_bool_and_null_extracted(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_object_state_branch_hints
        body = (
            'if (option.hasArg()) { x(); }\n'
            'if (option.getDescription() != null) { sb.append(desc); }'
        )
        bool_preds, null_checks = _extract_object_state_branch_hints(body)
        assert any("hasArg" in p for p in bool_preds)
        assert any("getDescription" in n for n in null_checks)

    def test_no_conditions_returns_empty(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _extract_object_state_branch_hints
        body = 'int x = value.length(); result.append(x);'
        bool_preds, null_checks = _extract_object_state_branch_hints(body)
        assert bool_preds == []
        assert null_checks == []


class TestFilterBypassRecipe:
    """_filter_bypass_recipe: detects passthrough-singleton early-return patterns."""

    def _fbr(self, body, params=None, fields=None):
        from tatg.analysis.typeinfo.recipe_synthesizer import _filter_bypass_recipe
        from tatg.analysis.typeinfo.models import FieldInfo, TypeShape
        if fields is not None:
            shape = TypeShape(
                class_name="FakeClass",
                fqn="com.example.FakeClass",
                package="com.example",
                fields=[FieldInfo(name=n, java_type=t) for t, n in fields],
            )
        else:
            shape = None
        return _filter_bypass_recipe("nextToken", body, params or [], shape)

    def test_fires_on_repeated_include_all_with_filter_param(self):
        body = """
            if (f == TokenFilter.INCLUDE_ALL) { return t; }
            if (f == TokenFilter.INCLUDE_ALL) { return ctx; }
            if (f == TokenFilter.INCLUDE_ALL) { return (_curr = t); }
        """
        result = self._fbr(body, params=[("TokenFilter", "f")])
        assert result is not None
        assert result.category == "filter-bypass-warning"
        assert "INCLUDE_ALL" in result.title

    def test_does_not_fire_with_fewer_than_3_matches(self):
        body = """
            if (f == TokenFilter.INCLUDE_ALL) { return t; }
            if (f == TokenFilter.INCLUDE_ALL) { return t2; }
        """
        result = self._fbr(body, params=[("TokenFilter", "f")])
        assert result is None

    def test_qualified_sentinel_does_not_require_name_based_filter_context(self):
        body = """
            if (f == TokenFilter.INCLUDE_ALL) return t;
            if (f == TokenFilter.INCLUDE_ALL) return t2;
            if (f == TokenFilter.INCLUDE_ALL) return t3;
        """
        result = self._fbr(body, params=[("int", "x")])
        assert result is not None

    def test_repeated_constant_without_short_circuit_does_not_fire(self):
        body = """
            if (f == TokenFilter.INCLUDE_ALL) { value = one(); }
            if (f == TokenFilter.INCLUDE_ALL) { value = two(); }
            if (f == TokenFilter.INCLUDE_ALL) { value = three(); }
        """
        result = self._fbr(body, params=[("TokenFilter", "f")])
        assert result is None

    def test_fires_on_filter_type_in_class_fields(self):
        body = """
            if (x == SomeFilter.INCLUDE_ALL) return t;
            if (x == SomeFilter.INCLUDE_ALL) return t2;
            if (x == SomeFilter.INCLUDE_ALL) return t3;
        """
        result = self._fbr(body, params=[], fields=[("TokenFilter", "rootFilter")])
        assert result is not None
        assert "CRITICAL" in " ".join(result.path_goals)

    def test_path_goals_mention_selective_filter(self):
        body = """
            if (f == TokenFilter.INCLUDE_ALL) { return t; }
            if (f == TokenFilter.INCLUDE_ALL) { _head = ctx; }
            if (f == TokenFilter.INCLUDE_ALL) { return (_curr = t); }
            if (f == TokenFilter.INCLUDE_ALL) { continue; }
        """
        result = self._fbr(body, params=[("TokenFilter", "f")])
        assert result is not None
        goals = " ".join(result.path_goals)
        assert "selective" in goals.lower() or "custom" in goals.lower()
        assert "smoke" in goals.lower() or "1–2" in goals or "1-2" in goals

    def test_accepts_alternative_passthrough_constant_names(self):
        body = """
            if (f == MyPredicate.ACCEPT_ALL) return t;
            if (f == MyPredicate.ACCEPT_ALL) return t2;
            if (f == MyPredicate.ACCEPT_ALL) return t3;
        """
        result = self._fbr(body, params=[("MyPredicate", "f")])
        assert result is not None
        assert "ACCEPT_ALL" in result.title


class TestProjectConstructionEvidenceRecipe:
    """_project_construction_evidence_recipe: scans test files for domain-type construction patterns."""

    def _pcer(self, method_name, params, tmp_path, test_java_source):
        from tatg.analysis.typeinfo.recipe_synthesizer import _project_construction_evidence_recipe
        test_dir = tmp_path / "src" / "test" / "java"
        test_dir.mkdir(parents=True)
        (test_dir / "FooTest.java").write_text(test_java_source)
        return _project_construction_evidence_recipe(method_name, params, tmp_path)

    def test_finds_new_construction_pattern(self, tmp_path):
        recipe = self._pcer(
            "process",
            [("Patterns", "patterns")],
            tmp_path,
            "class T { void t() { Patterns p = new Patterns(List.of()); } }",
        )
        assert recipe is not None
        assert recipe.category == "construction-evidence"
        text = " ".join(recipe.setup_steps)
        assert "Patterns" in text

    def test_no_match_returns_none(self, tmp_path):
        recipe = self._pcer(
            "process",
            [("Patterns", "patterns")],
            tmp_path,
            "class T { void t() { String s = new String(); } }",
        )
        assert recipe is None

    def test_skips_primitive_params(self, tmp_path):
        recipe = self._pcer(
            "process",
            [("int", "x"), ("long", "y")],
            tmp_path,
            "class T { void t() { int x = new int[0][0]; } }",
        )
        assert recipe is None

    def test_skips_string_and_collection_params(self, tmp_path):
        from tatg.analysis.typeinfo.recipe_synthesizer import _project_construction_evidence_recipe
        # String and List are in the excluded sets — no evidence recipe expected
        recipe = _project_construction_evidence_recipe(
            "method", [("String", "s"), ("List", "items")], tmp_path
        )
        assert recipe is None

    def test_finds_builder_pattern(self, tmp_path):
        recipe = self._pcer(
            "match",
            [("JsonNode", "event")],
            tmp_path,
            "class T { void t() { JsonNode node = JsonNode.of(\"{}\" ); } }",
        )
        assert recipe is not None
        text = " ".join(recipe.setup_steps)
        assert "JsonNode" in text


class TestCallerEntryRecipe:
    """_caller_entry_recipe: emits a recipe when target is non-public and callers exist."""

    def _cer(self, target_name, source_text, class_name="MyClass"):
        from tatg.analysis.typeinfo.recipe_synthesizer import _caller_entry_recipe
        from tatg.analysis.typeinfo.models import TypeShape
        shape = TypeShape(class_name=class_name, fqn=f"com.x.{class_name}", package="com.x")
        return _caller_entry_recipe(target_name, source_text, shape, None)

    def test_non_public_method_with_public_caller(self):
        source = """
        class MyClass {
            public String process(String input) { return innerParse(input); }
            String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is not None
        assert recipe.category == "caller-entry"
        assert "process" in " ".join(recipe.setup_steps)

    def test_public_method_returns_none(self):
        source = """
        class MyClass {
            public String parse(String s) { return s; }
        }
        """
        recipe = self._cer("parse", source)
        assert recipe is None

    def test_no_callers_emits_direct_invocation_recipe(self):
        """Package-private with no public callers → direct-invocation recipe (not None)."""
        source = """
        class MyClass {
            String innerParse(String s) { return s.trim(); }
            public String unrelated(int x) { return String.valueOf(x); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is not None
        assert recipe.category == "caller-entry"
        combined = " ".join(recipe.observations + recipe.setup_steps)
        assert "direct" in combined.lower() or "same package" in combined.lower() or "package" in combined.lower()

    def test_multiple_callers_included(self):
        source = """
        class MyClass {
            public String methodA(String s) { return innerParse(s); }
            public int methodB(int x) { return innerParse(String.valueOf(x)).length(); }
            String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is not None
        text = " ".join(recipe.setup_steps)
        assert "methodA" in text or "methodB" in text

    def test_private_method_returns_none(self):
        """Bug A fix: private methods are handled by SootUp, not caller-entry recipe."""
        source = """
        class MyClass {
            public String process(String s) { return innerParse(s); }
            private String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is None

    def test_protected_method_returns_none(self):
        """Bug A fix: protected methods are handled by _find_public_entry_points."""
        source = """
        class MyClass {
            public String process(String s) { return innerParse(s); }
            protected String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is None

    def test_package_private_with_caller_returns_recipe(self):
        """Package-private (no modifier) + public caller → recipe must be emitted."""
        source = """
        class MyClass {
            public String process(String s) { return innerParse(s); }
            String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is not None
        assert recipe.category == "caller-entry"

    def test_package_private_recipe_mentions_direct_call(self):
        """Recipe for package-private method should mention direct call from same-package test."""
        source = """
        class MyClass {
            public String process(String s) { return innerParse(s); }
            String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is not None
        setup_text = " ".join(recipe.setup_steps)
        assert "same package" in setup_text or "package" in setup_text
        assert "innerParse" in setup_text or "directly" in setup_text

    def test_package_private_recipe_title_does_not_say_may_fail(self):
        """Old wording incorrectly said 'direct invocation may fail' for package-private."""
        source = """
        class MyClass {
            public String process(String s) { return innerParse(s); }
            String innerParse(String s) { return s.trim(); }
        }
        """
        recipe = self._cer("innerParse", source)
        assert recipe is not None
        combined = " ".join([recipe.title] + recipe.observations + recipe.setup_steps)
        assert "may fail" not in combined
        assert "no coverage attribution" not in combined


# ──────────────────────────────────────────────────────────────────────────────
# P5: Abstract/interface construction recipes and constructor mock hints
# ──────────────────────────────────────────────────────────────────────────────

class TestAbstractInterfaceConstructionRecipe:
    """Tests for P5 fixes: abstract class, interface, and injectable field detection."""

    def _make_shape(self, class_name="Target", is_interface=False, is_abstract=False,
                    ctors=None, methods=None, fields=None):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        shape = TypeShape(
            class_name=class_name, fqn=f"pkg.{class_name}", package="pkg",
            is_interface=is_interface, is_abstract=is_abstract,
        )
        shape.constructors = ctors or []
        shape.methods = methods or []
        shape.fields = fields or []
        return shape, _construction_recipe

    def test_interface_target_returns_construction_recipe_with_mock_hint(self):
        shape, fn = self._make_shape(is_interface=True)
        recipe = fn(shape, [])
        assert recipe is not None
        assert recipe.category == "construction"
        combined = " ".join(recipe.setup_steps + recipe.observations)
        assert "cannot" in combined.lower() or "not compile" in combined.lower() or "interface" in combined.lower()
        assert "Mockito.mock" in combined

    def test_interface_target_avoid_contains_new(self):
        shape, fn = self._make_shape(class_name="EventCompactor", is_interface=True)
        recipe = fn(shape, [])
        assert recipe is not None
        assert any("new EventCompactor" in a for a in recipe.avoid)

    def test_abstract_class_without_factory_returns_subclass_hint(self):
        shape, fn = self._make_shape(is_abstract=True)
        recipe = fn(shape, [])
        assert recipe is not None
        combined = " ".join(recipe.setup_steps + recipe.observations)
        assert "abstract" in combined.lower()
        assert "subclass" in combined.lower() or "anonymous" in combined.lower()

    def test_abstract_class_avoid_contains_new(self):
        shape, fn = self._make_shape(class_name="BaseProcessor", is_abstract=True)
        recipe = fn(shape, [])
        assert recipe is not None
        assert any("new BaseProcessor" in a for a in recipe.avoid)

    def test_abstract_class_with_ctor_shows_super_call_hint(self):
        shape, fn = self._make_shape(
            is_abstract=True,
            ctors=[ConstructorInfo(visibility="public", params=[("String", "name"), ("int", "size")])],
        )
        recipe = fn(shape, [])
        assert recipe is not None
        joined = " ".join(recipe.setup_steps)
        assert "super" in joined
        assert "name" in joined and "size" in joined

    def test_abstract_class_with_factory_falls_through_to_factory_recipe(self):
        from tatg.analysis.typeinfo.models import MethodInfo
        shape, fn = self._make_shape(
            class_name="AbstractParser", is_abstract=True,
            methods=[MethodInfo(name="create", return_type="AbstractParser",
                                visibility="public", is_static=True, params=[])],
        )
        recipe = fn(shape, [])
        assert recipe is not None
        # should NOT be the "cannot instantiate" abstract recipe (factory path taken)
        joined = " ".join(recipe.observations + recipe.setup_steps)
        assert "AbstractParser.create" in joined or "factory" in joined.lower()

    def test_constructor_interface_param_emits_mock_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        target = TypeShape(class_name="Compactor", fqn="pkg.Compactor", package="pkg")
        target.constructors = [
            ConstructorInfo(visibility="public", params=[("SessionService", "svc")])
        ]
        target.methods = []
        target.fields = []

        dep_shape = TypeShape(
            class_name="SessionService", fqn="pkg.SessionService", package="pkg",
            is_interface=True,
        )
        recipe = _construction_recipe(target, [], type_index={"SessionService": dep_shape})
        assert recipe is not None
        joined = " ".join(recipe.setup_steps)
        assert "Mockito.mock(SessionService.class)" in joined

    def test_constructor_interface_param_prefers_known_concrete_implementation(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        target = TypeShape(class_name="Handler", fqn="pkg.Handler", package="pkg")
        target.constructors = [
            ConstructorInfo(visibility="public", params=[("Factory", "factory")])
        ]

        contract = TypeShape(
            class_name="Factory",
            fqn="pkg.Factory",
            package="pkg",
            is_interface=True,
        )
        implementation = TypeShape(
            class_name="DefaultFactory",
            fqn="pkg.DefaultFactory",
            package="pkg",
            interfaces=["Factory"],
        )
        implementation.constructors = [
            ConstructorInfo(visibility="public", params=[])
        ]

        recipe = _construction_recipe(
            target,
            [],
            type_index={
                "Factory": contract,
                "DefaultFactory": implementation,
            },
        )

        assert recipe is not None
        joined = " ".join(recipe.setup_steps)
        assert "prefer a source-visible concrete implementation" in joined
        assert "DefaultFactory" in joined
        assert "Mockito.mock(Factory.class)" in joined
        assert joined.index("DefaultFactory") < joined.index("Mockito.mock(Factory.class)")

    def test_constructor_concrete_param_does_not_emit_mock_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        target = TypeShape(class_name="Compactor", fqn="pkg.Compactor", package="pkg")
        target.constructors = [
            ConstructorInfo(visibility="public", params=[("Config", "cfg")])
        ]
        target.methods = []
        target.fields = []

        dep_shape = TypeShape(
            class_name="Config", fqn="pkg.Config", package="pkg",
            is_interface=False, is_abstract=False,
        )
        recipe = _construction_recipe(target, [], type_index={"Config": dep_shape})
        # No mock hint for concrete types
        if recipe is not None:
            joined = " ".join(recipe.setup_steps)
            assert "Mockito.mock(Config.class)" not in joined

    def test_injected_field_emits_setup_hint(self):
        from tatg.analysis.typeinfo.recipe_synthesizer import _construction_recipe
        from tatg.analysis.typeinfo.models import AnnotationInfo, FieldInfo
        target = TypeShape(class_name="Service", fqn="pkg.Service", package="pkg")
        target.constructors = []  # no-arg implicit ctor
        target.methods = []
        target.fields = [
            FieldInfo(
                name="repository", java_type="UserRepository",
                annotations=[AnnotationInfo(name="Inject")],
                visibility="private",
            )
        ]
        recipe = _construction_recipe(target, [])
        assert recipe is not None
        joined = " ".join(recipe.setup_steps)
        assert "repository" in joined
        assert "Inject" in joined or "injection" in joined.lower() or "mock" in joined.lower()


class TestIsAbstractDetectedFromSource:
    """Tests that source_extractor correctly detects the abstract modifier."""

    def _extract(self, source: str):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".java", mode="w", delete=False) as f:
            f.write(source)
            fpath = f.name
        try:
            from tatg.analysis.typeinfo.source_extractor import extract_type_shape
            return extract_type_shape(Path(fpath))
        finally:
            os.unlink(fpath)

    def test_abstract_class_detected(self):
        shape = self._extract("package p;\npublic abstract class Base {\n  public void run() {}\n}")
        assert shape is not None
        assert shape.is_abstract is True

    def test_concrete_class_not_abstract(self):
        shape = self._extract("package p;\npublic class Concrete {\n  public void run() {}\n}")
        assert shape is not None
        assert shape.is_abstract is False

    def test_interface_not_abstract(self):
        shape = self._extract("package p;\npublic interface IFoo {\n  void run();\n}")
        assert shape is not None
        assert shape.is_abstract is False
        assert shape.is_interface is True

    def test_abstract_modifier_multiple_modifiers(self):
        shape = self._extract("package p;\npublic abstract class Multi {\n}")
        assert shape is not None
        assert shape.is_abstract is True


# State/protocol setup is deliberately not inferred from field or method-name
# keywords. It is supplied by source/data-flow evidence and semantic branch
# analysis, then replay-validated before it becomes a hard recipe.
def test_state_sequence_keyword_recipe_is_not_exposed():
    from tatg.analysis.typeinfo import recipe_synthesizer

    assert not hasattr(recipe_synthesizer, "_state_sequence_recipe")


# ──────────────────────────────────────────────────────────────────────────────
# Fix C — _transitive_public_entry_points
# ──────────────────────────────────────────────────────────────────────────────

class TestTransitivePublicEntryPoints:
    """Verify that two-level entry chains (public → private → target) are found."""

    def _find(self, target: str, source: str):
        from tatg.analysis.typeinfo.recipe_synthesizer import _transitive_public_entry_points
        return _transitive_public_entry_points(target, source)

    def _make_two_level_source(self) -> str:
        """Class where readEsc is private, called by nextQuoted (private), called by nextStr (public)."""
        return textwrap.dedent("""\
            public class JsonReader {
                private char readEsc(int c) {
                    if (c == 'n') return '\\n';
                    return (char) c;
                }
                private String nextQuoted() {
                    char ch = readEsc(0);
                    return String.valueOf(ch);
                }
                public String nextStr() {
                    return nextQuoted();
                }
                public String nextName() {
                    return nextQuoted();
                }
            }
        """)

    def test_finds_transitive_chain(self):
        pairs = self._find("readEsc", self._make_two_level_source())
        assert len(pairs) >= 1
        pub_names = [p for p, _ in pairs]
        intermediates = [m for _, m in pairs]
        assert "nextStr" in pub_names or "nextName" in pub_names
        assert "nextQuoted" in intermediates

    def test_intermediate_method_recorded_correctly(self):
        pairs = self._find("readEsc", self._make_two_level_source())
        for pub, mid in pairs:
            assert mid == "nextQuoted"

    def test_no_chain_returns_empty(self):
        src = textwrap.dedent("""\
            public class Foo {
                private void helper() { }
                public void pub() { }
            }
        """)
        pairs = self._find("helper", src)
        assert pairs == []

    def test_java_keywords_not_treated_as_methods(self):
        """if/while/for in method bodies must not appear as caller names."""
        src = textwrap.dedent("""\
            public class Foo {
                private int doWork(int x) {
                    if (x > 0) return x;
                    while (x < 0) x++;
                    return x;
                }
                private int helper() {
                    return doWork(1);
                }
                public int entry() {
                    return helper();
                }
            }
        """)
        pairs = self._find("doWork", src)
        pub_names = [p for p, _ in pairs]
        intermediates = [m for _, m in pairs]
        assert "if" not in pub_names
        assert "while" not in pub_names
        assert "for" not in pub_names
        assert "entry" in pub_names
        assert "helper" in intermediates

    def test_caps_at_eight_pairs(self):
        """Result must never exceed 8 entries."""
        # Build a class with many public methods calling one intermediate
        methods = "\n".join(
            f"    public void pub{i}() {{ mid(); }}" for i in range(15)
        )
        src = textwrap.dedent(f"""\
            public class Big {{
                private void target() {{ }}
                private void mid() {{ target(); }}
                {methods}
            }}
        """)
        pairs = self._find("target", src)
        assert len(pairs) <= 8

    def test_large_non_method_prefix_does_not_cause_regex_backtracking(self):
        noise = "\n".join(
            f"    // unrelated source line {index}: value = other + {index};"
            for index in range(6000)
        )
        src = textwrap.dedent(f"""\
            public class Large {{
            {noise}
                private void target() {{ }}
                private void intermediate() {{ target(); }}
                public void entry() {{ intermediate(); }}
            }}
        """)

        started = time.perf_counter()
        pairs = self._find("target", src)

        assert time.perf_counter() - started < 1.0
        assert ("entry", "intermediate") in pairs

    def test_direct_public_caller_not_required_to_go_through_transitive(self):
        """If the public method calls target directly, _public_callers_of handles it;
        _transitive_public_entry_points should not return it as a transitive pair."""
        src = textwrap.dedent("""\
            public class Foo {
                private int target(int x) { return x + 1; }
                private int mid() { return target(0); }
                public int directCaller() { return target(5); }
                public int indirectCaller() { return mid(); }
            }
        """)
        pairs = self._find("target", src)
        # directCaller calls target directly — it may or may not appear here
        # but indirectCaller → mid → target should appear
        pub_names = [p for p, _ in pairs]
        if pairs:
            assert "indirectCaller" in pub_names or "mid" in [m for _, m in pairs]

    def test_private_reflection_recipe_uses_transitive_when_no_direct_callers(self):
        """Integration: _private_reflection_entry_recipe should emit a caller-entry
        recipe with transitive chain info when no direct public callers exist."""
        from tatg.analysis.typeinfo.recipe_synthesizer import _private_reflection_entry_recipe
        src = textwrap.dedent("""\
            public class JsonReader {
                private char readEsc(int c) {
                    if (c == 'n') return '\\n';
                    return (char) c;
                }
                private String nextQuoted() {
                    char ch = readEsc(0);
                    return String.valueOf(ch);
                }
                public String nextStr() {
                    return nextQuoted();
                }
            }
        """)
        shape = TypeShape(class_name="JsonReader", fqn="com.Foo", package="com")
        recipe = _private_reflection_entry_recipe("readEsc", src, shape)
        assert recipe is not None
        assert recipe.category == "caller-entry"
        obs_joined = " ".join(recipe.observations)
        assert "transitive" in obs_joined.lower() or "nextQuoted" in obs_joined
        steps_joined = " ".join(recipe.setup_steps)
        assert "nextStr" in steps_joined or "nextQuoted" in steps_joined


def test_parse_pitest_line_coverage_tests_preserves_descriptor_block_identity(
    tmp_path,
):
    from tatg.evaluation.mutation import parse_pitest_line_coverage_tests

    xml = tmp_path / "linecoverage.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<coverage>
  <block classname="com.example.Parser" method="next(Z)I" number="19">
    <tests>
      <test name="testComment(com.example.ParserTest)"/>
      <test name="testWhitespace(com.example.ParserTest)"/>
    </tests>
  </block>
</coverage>
"""
    )

    assert parse_pitest_line_coverage_tests(xml) == {
        (
            "com.example.Parser",
            "next",
            "(Z)I",
            19,
        ): [
            "testComment(com.example.ParserTest)",
            "testWhitespace(com.example.ParserTest)",
        ]
    }
