package com.acme;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int abs(int value) {
        if (value < 0) {
            return -value;
        }
        return value;
    }
}
