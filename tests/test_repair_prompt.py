from tatg.repair import _unresolved_symbol_guidance


def test_unresolved_symbol_guidance_includes_declared_type_instance_call():
    output = """
[ERROR] /tmp/FooTest.java:[21,37] cannot find symbol
[ERROR]   symbol:   method createWriter(java.io.ByteArrayOutputStream)
[ERROR]   location: variable factory of type com.acme.XmlFactory
""".strip()

    guidance = _unresolved_symbol_guidance(output)

    assert "method createWriter(java.io.ByteArrayOutputStream)" in guidance
    assert "Unavailable instance calls observed:" in guidance
    assert "factory.createWriter(java.io.ByteArrayOutputStream) is not declared on com.acme.XmlFactory" in guidance
    assert "repair that construction path" in guidance


def test_unresolved_symbol_guidance_handles_truncated_maven_tail():
    output = """
pilation failure
[ERROR]   symbol:   method createWriter(java.io.ByteArrayOutputStream)
[ERROR]   location: variable factory of type com.acme.XmlFactory
""".strip()

    guidance = _unresolved_symbol_guidance(output)

    assert "factory.createWriter(java.io.ByteArrayOutputStream) is not declared on com.acme.XmlFactory" in guidance


def test_unresolved_symbol_guidance_includes_incomplete_abstract_implementation():
    output = """
[ERROR] cannot find symbol
[ERROR] <anonymous com.acme.FooTest$1> is not abstract and does not override abstract method putBytes(java.nio.ByteBuffer) in com.acme.Payload
""".strip()

    guidance = _unresolved_symbol_guidance(output)

    assert "Incomplete abstract/interface implementations observed:" in guidance
    assert "com.acme.Payload missed putBytes(java.nio.ByteBuffer)" in guidance
    assert "focused mock" in guidance


def test_unresolved_symbol_guidance_includes_class_and_static_context_blockers():
    output = """
[ERROR] /tmp/FooTest.java:[12,40] cannot find symbol
[ERROR]   symbol:   method text(java.lang.String)
[ERROR]   location: class io.acme.ChatResponse.Builder
[ERROR] /tmp/FooTest.java:[18,38] non-static method createParser(java.lang.String) cannot be referenced from a static context
""".strip()

    guidance = _unresolved_symbol_guidance(output)

    assert "io.acme.ChatResponse.Builder.text(java.lang.String) is not declared" in guidance
    assert "createParser(java.lang.String) is an instance method used from a static context" in guidance
