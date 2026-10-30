package org.example;

import java.util.*;
import java.util.concurrent.*;

/**
 * Manages parallel execution of callgraph path searches.
 * Encapsulates thread pool lifecycle and synchronization logic.
 */
public class ParallelSearchManager {

    private final int threadCount;
    private final ExecutorService executor;
    private final Object statsLock = new Object();

    public ParallelSearchManager(int threadCount) {
        this.threadCount = Math.max(1, threadCount);
        this.executor = threadCount > 1 
                ? Executors.newFixedThreadPool(this.threadCount)
                : null;
    }

    /**
     * Executes multiple search tasks in parallel (if threadCount > 1).
     * Aggregates results safely from all tasks.
     */
    public void executeParallel(List<SearchTask> tasks,
                                 CallGraphPathFinder.SearchStats aggregatedStats) throws InterruptedException {
        if (threadCount == 1 || tasks.isEmpty()) {
            // Single-threaded fallback
            for (SearchTask task : tasks) {
                task.run();
            }
            return;
        }

        // Parallel execution with thread pool
        List<Future<?>> futures = new ArrayList<>();
        for (SearchTask task : tasks) {
            futures.add(executor.submit(task));
        }

        // Wait for all tasks to complete
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (ExecutionException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Search task failed: " + e.getCause(), e.getCause());
            }
        }

        // Aggregate statistics from all tasks
        for (SearchTask task : tasks) {
            synchronized (statsLock) {
                aggregatedStats.exploredStates += task.getStats().exploredStates;
                aggregatedStats.emittedPathCandidates += task.getStats().emittedPathCandidates;
                if (task.getStats().truncatedByPathLimit) {
                    aggregatedStats.truncatedByPathLimit = true;
                }
                if (task.getStats().truncatedByStateLimit) {
                    aggregatedStats.truncatedByStateLimit = true;
                }
                if (task.getStats().stopRequested) {
                    aggregatedStats.stopRequested = true;
                }
            }
        }
    }

    /**
     * Creates a synchronized consumer that safely updates shared path collection.
     */
    public static <T> java.util.function.Consumer<T> synchronizedConsumer(
            Object lock,
            java.util.function.Consumer<T> delegate) {
        return item -> {
            synchronized (lock) {
                delegate.accept(item);
            }
        };
    }

    /**
     * Shuts down the executor service gracefully.
     */
    public void shutdown() {
        if (executor != null) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Holder for a single search task and its statistics.
     */
    public static class SearchTask implements Runnable {
        private final Runnable taskLogic;
        private final CallGraphPathFinder.SearchStats stats;

        public SearchTask(Runnable taskLogic, CallGraphPathFinder.SearchStats stats) {
            this.taskLogic = taskLogic;
            this.stats = stats;
        }

        @Override
        public void run() {
            taskLogic.run();
        }

        public CallGraphPathFinder.SearchStats getStats() {
            return stats;
        }
    }
}
