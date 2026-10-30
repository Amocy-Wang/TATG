package com.acme;

import org.junit.Test;

public class CalculatorTest {

@Test
public void add_returnsSumAfterFallback() {
    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));
}
}

