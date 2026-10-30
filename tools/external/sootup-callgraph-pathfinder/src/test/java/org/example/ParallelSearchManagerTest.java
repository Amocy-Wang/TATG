package org.example;

import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ParallelSearchManagerTest {

    @Test
    void singleThreadedModeRunsAllTasks() {
        AtomicInteger counter = new AtomicInteger(0);
        List<ParallelSearchManager.SearchTask> tasks = createIncrementingTasks(5, counter);

        ParallelSearchManager mgr = new ParallelSearchManager(1);
        try {
            CallGraphPathFinder.SearchStats agg = new CallGraphPathFinder.SearchStats();
            mgr.executeParallel(tasks, agg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Interrupted");
        } finally {
            mgr.shutdown();
        }

        assertEquals(5, counter.get());
    }

    @Test
    void multiThreadedModeRunsAllTasks() {
        AtomicInteger counter = new AtomicInteger(0);
        List<ParallelSearchManager.SearchTask> tasks = createIncrementingTasks(10, counter);

        ParallelSearchManager mgr = new ParallelSearchManager(4);
        try {
            CallGraphPathFinder.SearchStats agg = new CallGraphPathFinder.SearchStats();
            mgr.executeParallel(tasks, agg);
            assertEquals(10, counter.get());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Interrupted");
        } finally {
            mgr.shutdown();
        }
    }

    @Test
    void aggregatesExploredStatesFromAllTasks() {
        List<ParallelSearchManager.SearchTask> tasks = new ArrayList<>();
        for (int i = 1; i <= 4; i++) {
            final long states = i * 100L;
            CallGraphPathFinder.SearchStats taskStats = new CallGraphPathFinder.SearchStats();
            taskStats.exploredStates = states;
            tasks.add(new ParallelSearchManager.SearchTask(() -> {}, taskStats));
        }

        ParallelSearchManager mgr = new ParallelSearchManager(2);
        CallGraphPathFinder.SearchStats agg = new CallGraphPathFinder.SearchStats();
        try {
            mgr.executeParallel(tasks, agg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Interrupted");
        } finally {
            mgr.shutdown();
        }

        // 1*100 + 2*100 + 3*100 + 4*100 = 1000
        assertEquals(1000L, agg.exploredStates);
    }

    @Test
    void aggregatesTruncationFlagsFromAllTasks() {
        CallGraphPathFinder.SearchStats statsA = new CallGraphPathFinder.SearchStats();
        statsA.truncatedByPathLimit = true;
        CallGraphPathFinder.SearchStats statsB = new CallGraphPathFinder.SearchStats();
        statsB.truncatedByStateLimit = true;
        CallGraphPathFinder.SearchStats statsC = new CallGraphPathFinder.SearchStats();

        List<ParallelSearchManager.SearchTask> tasks = List.of(
                new ParallelSearchManager.SearchTask(() -> {}, statsA),
                new ParallelSearchManager.SearchTask(() -> {}, statsB),
                new ParallelSearchManager.SearchTask(() -> {}, statsC)
        );

        ParallelSearchManager mgr = new ParallelSearchManager(3);
        CallGraphPathFinder.SearchStats agg = new CallGraphPathFinder.SearchStats();
        try {
            mgr.executeParallel(tasks, agg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Interrupted");
        } finally {
            mgr.shutdown();
        }

        assertTrue(agg.truncatedByPathLimit, "Should aggregate truncatedByPathLimit");
        assertTrue(agg.truncatedByStateLimit, "Should aggregate truncatedByStateLimit");
    }

    @Test
    void emptyTaskListCompletes() {
        ParallelSearchManager mgr = new ParallelSearchManager(4);
        try {
            CallGraphPathFinder.SearchStats agg = new CallGraphPathFinder.SearchStats();
            assertDoesNotThrow(() -> mgr.executeParallel(Collections.emptyList(), agg));
            assertEquals(0L, agg.exploredStates);
        } finally {
            mgr.shutdown();
        }
    }

    @Test
    void threadCountLessThanOneIsNormalizedToOne() {
        // Should not throw; ParallelSearchManager normalizes <= 0 to 1
        ParallelSearchManager mgr = new ParallelSearchManager(-3);
        AtomicInteger counter = new AtomicInteger(0);
        List<ParallelSearchManager.SearchTask> tasks = createIncrementingTasks(3, counter);

        try {
            CallGraphPathFinder.SearchStats agg = new CallGraphPathFinder.SearchStats();
            mgr.executeParallel(tasks, agg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Interrupted");
        } finally {
            mgr.shutdown();
        }

        assertEquals(3, counter.get());
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private List<ParallelSearchManager.SearchTask> createIncrementingTasks(
            int n, AtomicInteger counter) {
        List<ParallelSearchManager.SearchTask> tasks = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            CallGraphPathFinder.SearchStats stats = new CallGraphPathFinder.SearchStats();
            stats.exploredStates = 1;
            tasks.add(new ParallelSearchManager.SearchTask(counter::incrementAndGet, stats));
        }
        return tasks;
    }
}
