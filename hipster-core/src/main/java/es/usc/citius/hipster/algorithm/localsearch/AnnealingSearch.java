package es.usc.citius.hipster.algorithm.localsearch;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.security.SecureRandom;

import es.usc.citius.hipster.algorithm.Algorithm;
import es.usc.citius.hipster.model.HeuristicNode;
import es.usc.citius.hipster.model.Node;
import es.usc.citius.hipster.model.function.NodeExpander;

/**
 * Implementation of the simulated annealing search that is a probabilistic
 * technique for approximating the global optimum of a given function.
 */
public class AnnealingSearch<A, S, N extends HeuristicNode<A, S, Double, N>> extends Algorithm<A, S, N> {

    static final private Double DEFAULT_ALPHA = 0.9;
    static final private Double DEFAULT_MIN_TEMP = 0.00001;
    static final private Double START_TEMP = 1.;

    private N initialNode;
    private Double alpha;
    private Double minTemp;
    private AcceptanceProbability acceptanceProbability;
    private SuccessorFinder<A, S, N> successorFinder;
    private NodeExpander<A, S, N> nodeExpander;

    private final SecureRandom randIndGen = new SecureRandom();

    public AnnealingSearch(N initialNode, NodeExpander<A, S, N> nodeExpander, Double alpha, Double minTemp,
            AcceptanceProbability acceptanceProbability, SuccessorFinder<A, S, N> successorFinder) {

        if (initialNode == null) {
            throw new IllegalArgumentException("Provide a valid initial node");
        }
        this.initialNode = initialNode;

        if (nodeExpander == null) {
            throw new IllegalArgumentException("Provide a valid node expander");
        }
        this.nodeExpander = nodeExpander;

        this.alpha = (alpha != null && alpha > 0. && alpha < 1.0) ? alpha : DEFAULT_ALPHA;
        this.minTemp = (minTemp != null && minTemp >= 0. && minTemp <= 1.0) ? minTemp : DEFAULT_MIN_TEMP;

        this.acceptanceProbability = (acceptanceProbability != null) ? acceptanceProbability : new AcceptanceProbability() {
            @Override
            public Double compute(Double oldScore, Double newScore, Double temp) {
                return (newScore < oldScore ? 1 : Math.exp((oldScore - newScore) / temp));
            }
        };

        this.successorFinder = (successorFinder != null) ? successorFinder : new SuccessorFinder<A, S, N>() {
            @Override
            public N estimate(N node, NodeExpander<A, S, N> nodeExpander) {
                List<N> successors = new ArrayList<>();
                for (N successor : nodeExpander.expand(node)) {
                    successors.add(successor);
                }
                // --- REUTILIZA randIndGen ---
                return successors.get(Math.abs(randIndGen.nextInt()) % successors.size());
            }
        };
    }

    @Override
    public ASIterator iterator() {
        return new ASIterator();
    }

    public class ASIterator implements Iterator<N> {

        private Queue<N> queue = new LinkedList<>();
        private Double bestScore = null;
        private Double curTemp = START_TEMP;

        private ASIterator() {
            bestScore = initialNode.getEstimation();
            queue.add(initialNode);
        }

        @Override
        public boolean hasNext() {
            return !queue.isEmpty();
        }

        @Override
        public N next() {
            N currentNode = this.queue.poll();
            if (curTemp > minTemp) {
                N newNode = null;
                for (int i = 0; i < 100; i++) {
                    N randSuccessor = successorFinder.estimate(currentNode, nodeExpander);
                    Double score = randSuccessor.getScore();
                    // --- REUTILIZA randIndGen ---
                    if (acceptanceProbability.compute(bestScore, score, curTemp) > randIndGen.nextDouble()) {
                        newNode = randSuccessor;
                        bestScore = score;
                    }
                }
                if (newNode != null) {
                    queue.add(newNode);
                } else {
                    queue.add(currentNode);
                }
                curTemp *= alpha;
            }
            return currentNode;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    public interface AcceptanceProbability {
        Double compute(Double oldScore, Double newScore, Double temp);
    }

    public interface SuccessorFinder<A, S, N extends Node<A, S, N>> {
        N estimate(N node, NodeExpander<A, S, N> nodeExpander);
    }
}
