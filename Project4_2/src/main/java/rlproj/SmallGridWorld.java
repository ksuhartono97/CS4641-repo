package rlproj;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.debugtools.DPrint;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.Iterator;
import java.util.List;
import java.util.Set;


public class SmallGridWorld {
    GridWorldDomain gwdg;
    OOSADomain domain;
    RewardFunction rf;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;

    protected int goalx = 10;
    protected int goaly = 10;


    protected int heightx = 11;
    protected int heighty = 11;

    // Each inner array is a column (y)
    //Rightmost is bottommost
    protected int [][] map = new int[][]{
            {0,0,0,0,0,1,0,0,0,1,0},
            {0,0,0,1,0,0,0,1,0,0,0},
            {0,0,0,1,1,1,0,0,0,1,0},
            {0,0,0,0,0,1,0,0,0,0,0},
            {0,0,0,0,0,1,0,0,0,0,0},
            {1,0,1,1,1,1,1,1,0,1,1},
            {0,0,1,0,1,0,0,1,0,0,0},
            {0,0,0,0,1,0,0,1,1,0,0},
            {0,0,1,0,0,0,0,0,0,0,0},
            {0,0,0,0,1,0,0,0,0,0,0},
            {0,0,0,0,1,0,0,0,0,0,0}
    };

    protected int [][] rewardMap = new int[][]{
            {0,0,0,0,0,1,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,1,0,0,0,0,0},
            {0,0,0,0,0,1,0,0,0,0,0},
            {0,0,0,0,0,1,0,0,0,0,0},
            {1,0,1,1,1,1,1,1,0,1,1},
            {0,0,0,0,1,0,0,0,0,0,0},
            {0,0,0,0,1,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,1,0,0,0,0,0,0},
            {0,0,0,0,1,0,0,0,0,0,0}
    };

    public class CustomValueIteration extends ValueIteration {
        protected int totalValueIterations;

        public CustomValueIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxIterations) {
            super(domain, gamma, hashingFactory, maxDelta, maxIterations);
            totalValueIterations = 0;
        }

        public int getTotalValueIterations() {
            return totalValueIterations;
        }

        @Override
        public void runVI() {
            if (!this.foundReachableStates) {
                throw new RuntimeException("Cannot run VI until the reachable states have been found. Use the planFromState or performReachabilityFrom method at least once before calling runVI.");
            } else {
                Set<HashableState> states = this.valueFunction.keySet();

                int i;
                for(i = 0; i < this.maxIterations; ++i) {
                    double delta = 0.0D;


                    double v;
                    double maxQ;
                    for(Iterator var5 = states.iterator(); var5.hasNext(); delta = Math.max(Math.abs(maxQ - v), delta)) {
                        HashableState sh = (HashableState)var5.next();
                        v = this.value(sh);
                        maxQ = this.performBellmanUpdateOn(sh);
                    }

                    if (delta < this.maxDelta) {
                        break;
                    }
                }

                this.totalValueIterations += i;

                DPrint.cl(this.debugCode, "Passes: " + i);
                DPrint.cl(this.debugCode, "Total iterations: " + this.getTotalValueIterations());
                this.hasRunVI = true;
            }
        }
    }

    public SmallGridWorld(){
        gwdg = new GridWorldDomain(heightx, heighty);
        gwdg.setMap(map);
        tf = new GridWorldTerminalFunction(this.goalx, this.goaly);
        gwdg.setTf(tf);
        GridWorldRewardFunction tempRf = new GridWorldRewardFunction(heightx,heighty, -1.0);
        tempRf.setReward(10, 10, 10);
        tempRf.setReward(1, 5, -3);
        gwdg.setRf(tempRf);
        goalCondition = new TFGoalCondition(tf);
        gwdg.setProbSucceedTransitionDynamics(0.3);
        domain = gwdg.generateDomain();

        initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);
    }


    public void visualize(String outputPath){
        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
        new EpisodeSequenceVisualizer(v, domain, outputPath);
    }

    public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

        List<State> allStates = StateReachability.getReachableStates(initialState,
                domain, hashingFactory);
        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, heightx, heighty, valueFunction, p);
        gui.initGUI();

    }

//    public void valueIteration(String outputPath){
//
//        Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100);
//        Policy p = planner.planFromState(initialState);
//
//        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");
//
//    }

    public void valueIteration(String outputPath){

        Planner planner = new ValueIteration(domain, 0.6, hashingFactory, 0.001, 100);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

        simpleValueFunctionVis((ValueFunction)planner, p);
//        p.getTotalValueIterations
//        System.out.println(planner.getTOtal)

    }

    public void policyIteration(String outputPath){

        Planner planner = new PolicyIteration(domain, 0.9, hashingFactory, 0.001, 100, 100);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");

        simpleValueFunctionVis((ValueFunction)planner, p);
    }


    public void setGoalLocation(int goalx, int goaly){
        this.goalx = goalx;
        this.goaly = goaly;
    }

    public static void main(String[] args) {
        SmallGridWorld example = new SmallGridWorld();
        String outputPath = "output/"; //directory to record results

        //run example
        example.policyIteration(outputPath);

        //run the visualizer
        example.visualize(outputPath);
    }


}